# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Speech-to-speech conversation bot."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner, PipelineMetadata

# Uncomment the below lines enable speculative speech processing
# from nvidia_pipecat.processors.nvidia_context_aggregator import (
#     NvidiaTTSResponseCacher,
#     create_nvidia_context_aggregator,
# )
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService
from nvidia_pipecat.transports.network.ace_fastapi_websocket import ACETransport, ACETransportParams
from nvidia_pipecat.transports.services.ace_controller.routers.websocket_router import router as websocket_router
from nvidia_pipecat.utils.logging import setup_default_ace_logging

load_dotenv(override=True)

setup_default_ace_logging(level="DEBUG")


async def create_pipeline_task(pipeline_metadata: PipelineMetadata):
    """Create the pipeline to be run.

    Args:
        pipeline_metadata (PipelineMetadata): Metadata containing websocket and other pipeline configuration.

    Returns:
        PipelineTask: The configured pipeline task for handling speech-to-speech conversation.
    """
    transport = ACETransport(
        websocket=pipeline_metadata.websocket,
        params=ACETransportParams(
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=20,
        ),
    )

    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY", "not-needed-for-local-nim"),
        base_url=os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct"),
    )

    stt = RivaASRService(
        server=os.getenv("RIVA_ASR_URL", "localhost:50051"),
        api_key=os.getenv("NVIDIA_API_KEY", "not-needed-for-local-nim"),
        language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
        sample_rate=16000,
        model=os.getenv("RIVA_ASR_MODEL", "parakeet-1.1b-en-US-asr-streaming-silero-vad-asr-bls-ensemble"),
    )

    tts = RivaTTSService(
        server=os.getenv("RIVA_TTS_URL", "localhost:50051"),
        api_key=os.getenv("NVIDIA_API_KEY", "not-needed-for-local-nim"),
        voice_id=os.getenv("RIVA_TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Sofia"),
        model=os.getenv("RIVA_TTS_MODEL", "magpie_tts_ensemble-Magpie-Multilingual"),
        language=os.getenv("RIVA_TTS_LANGUAGE", "en-US"),
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
    )

    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

    # System prompt can be changed to fit the use case
    messages = [
        {
            "role": "system",
            "content": (
                "### CONVERSATION CONSTRAINTS\n"
                "STRICTLY answer in 1-2 sentences or less than 200 characters. "
                "This must be followed very rigorously; it is crucial.\n"
                "Output must be plain text, unformatted, and without any special characters - "
                "suitable for direct conversion to speech.\n"
                "DO NOT use bullet points, lists, code samples, or headers in your spoken responses.\n"
                "STRICTLY be short, concise, and to the point. Avoid elaboration, explanation, or repetition.\n"
                "Pronounce numbers, dates, and special terms. For phone numbers, read digits slowly and separately. "
                "For times, use natural phrasing like 'seven o'clock a.m.' instead of 'seven zero zero.'\n"
                "Silently correct likely transcription errors by inferring the intended meaning without saying "
                "`did you mean..` or `I think you meant..`. "
                "Prioritize what the user meant, not just the literal words.\n"
                "### OPENING PROTOCOL\n"
                "STRICTLY START CONVERSATION WITH 'Thank you for calling GreenForce Garden. "
                "What can I do for you today?'\n"
                "### CLOSING PROTOCOL\n"
                "End with either 'Have a green day!' or 'Have a good one.' Use one consistently per call.\n"
                "### YOU ARE ...\n"
                "You are Flora, the voice of 'GreenForce Garden', a San Francisco flower shop "
                "powered by NVIDIA GPUs.\n"
                "You're cool, upbeat, and love making people smile with your floral know-how.\n"
                "You embody warmth, expertise, and dedication to creating a perfect floral experience.\n"
                "### CONVERSATION GUIDELINES\n"
                "CORE RESPONSIBILITIES - Order Management, Consultation, Inventory Guidance, "
                "Delivery Coordination, Customer Care, Giving Fun Advice\n"
                "While taking orders, have occasion understanding, ask for recipient details, "
                "customer preferences, and delivery planning\n"
                "SUGGEST cards with personal messages\n"
                "SUGGEST seasonal recommendations (e.g., spring: tulips, pastels; romance: roses, peonies) "
                "and occasion-specific details (e.g., elegant wrapping).\n"
                "SUGGEST complementary items: vases, chocolates, cards. "
                "Also provide care instructions for long-lasting enjoyment.\n"
                "STRICTLY Confirm all order details before finalizing: flowers, colors, "
                "delivery address, timing\n"
                "STRICTLY Collect complete contact information for order updates\n"
                "STRICTLY Provide ORDER CONFIRMATION with ESTIMATED DELIVERY TIMES\n"
                "OFFER MULTIPLE PAYMENT OPTIONS (e.g., card, cash, online) and confirm SECURE PROCESSING.\n"
                "STRICTLY If you are unsure about a request, ask clarifying questions "
                "to ensure you understand before responding."
            ),
        },
    ]

    context = OpenAILLMContext(messages)

    # Comment out the below line when enabling Speculative Speech Processing
    context_aggregator = llm.create_context_aggregator(context)

    # Uncomment the below line to enable speculative speech processing
    # nvidia_context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
    # Uncomment the below line to enable speculative speech processing
    # nvidia_tts_response_cacher = NvidiaTTSResponseCacher()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            stt_transcript_synchronization,
            # Comment out the below line when enabling Speculative Speech Processing
            context_aggregator.user(),
            # Uncomment the below line to enable speculative speech processing
            # nvidia_context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            # Caches TTS responses for coordinated delivery in speculative
            # speech processing
            # nvidia_tts_response_cacher,  # Uncomment to enable speculative speech processing
            tts_transcript_synchronization,
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
            # Uncomment the below line to enable speculative speech processing
            # nvidia_context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=True,
            start_metadata={"stream_id": pipeline_metadata.stream_id},
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    return task


app = FastAPI()
app.include_router(websocket_router)
runner = ACEPipelineRunner.create_instance(pipeline_callback=create_pipeline_task)
app.mount("/static", StaticFiles(directory=os.getenv("STATIC_DIR", "../static")), name="static")

if __name__ == "__main__":
    uvicorn.run("bot:app", host="0.0.0.0", port=8100, workers=4)
