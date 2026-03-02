# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Speech-to-speech conversation bot."""

import os
from enum import Enum
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from nvidia_pipecat.services.riva_speech import NemotronASRService, NemotronTTSService
from nvidia_pipecat.utils.logging import setup_default_logging

load_dotenv(override=True)


class VADProfile(Enum):
    """VAD Profile options."""

    SILERO = "Silero"  # Transport Silero VAD analyzer
    ASR = "ASR"  # ASR VAD


VAD_PROFILE = VADProfile(os.getenv("VAD_PROFILE", VADProfile.ASR))

setup_default_logging(level="DEBUG", exclude_warning=True)


async def run_bot(websocket: WebSocket, stream_id: str):
    """Run the voice agent bot with WebSocket.

    Args:
        websocket (WebSocket): The WebSocket connection for audio streaming.
        stream_id (str): The unique identifier for the conversation stream.
    """
    # Initialize WebSocket transport with protobuf serialization
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=10,
            add_wav_header=True,
            serializer=ProtobufFrameSerializer(),
            vad_analyzer=SileroVADAnalyzer() if VAD_PROFILE == VADProfile.SILERO else None,
        ),
    )

    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct"),
    )

    stt = NemotronASRService(
        server=os.getenv("ASR_SERVER_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        language=os.getenv("ASR_LANGUAGE", "en-US"),
        sample_rate=16000,
        model=os.getenv("ASR_MODEL_NAME", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"),
        generate_interruptions=VAD_PROFILE == VADProfile.ASR,
    )

    tts = NemotronTTSService(
        server=os.getenv("TTS_SERVER_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id=os.getenv("TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Aria"),
        model=os.getenv("TTS_MODEL_NAME", "magpie_tts_ensemble-Magpie-Multilingual"),
        language=os.getenv("TTS_LANGUAGE", "en-US"),
        sample_rate=16000,
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
    )

    # System prompt can be changed to fit the use case
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always answer as helpful, friendly, and polite. "
            "Respond with one sentence or less than 75 characters. "
            "Do not respond with bulleted or numbered list. ",
        },
    ]

    context = LLMContext(messages)

    # Configure speculative speech processing based on environment variable
    enable_speculative_speech = os.getenv("ENABLE_SPECULATIVE_SPEECH", "true").lower() == "true"
    raw_chat_history = os.getenv("CHAT_HISTORY_LIMIT")
    try:
        chat_history_limit = int(raw_chat_history) if raw_chat_history is not None else 20
    except ValueError:
        logger.warning(f"Invalid CHAT_HISTORY_LIMIT {raw_chat_history!r}, falling back to default 20")
        chat_history_limit = 20

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(
            context, send_interims=True, chat_history_limit=chat_history_limit
        )
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = create_nvidia_context_aggregator(
            context, send_interims=False, chat_history_limit=chat_history_limit
        )
        tts_response_cacher = None

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            *([tts_response_cacher] if tts_response_cacher else []),  # Include cacher only if enabled
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=True,
            start_metadata={"stream_id": stream_id},
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesUpdateFrame(messages=messages, run_llm=True)])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


app = FastAPI()


@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    """Accept the WebSocket connection and run the pipeline.

    Args:
        websocket (WebSocket): The WebSocket connection.
        stream_id (str): The ID of the stream.
    """
    await websocket.accept()
    logger.info(f"Accepted WebSocket connection for stream ID {stream_id}")
    try:
        with logger.contextualize(stream_id=stream_id):
            await run_bot(websocket, stream_id)
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        await websocket.close(code=1000, reason="Internal Server Error")


app.mount("/static", StaticFiles(directory=os.getenv("STATIC_DIR", "./static")), name="static")

if __name__ == "__main__":
    uvicorn.run("bot:app", host="0.0.0.0", port=8100, workers=4)
