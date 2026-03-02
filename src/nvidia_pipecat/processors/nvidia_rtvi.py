# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA RTVI Processors.

This module provides RTVI protocol processors for handling NVIDIA Nemotron Speech ASR and TTS Services specific
client messages and outputting frames to the client.
"""

import base64
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import BotStoppedSpeakingFrame, EndFrame, Frame, LLMMessagesUpdateFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIClientMessage, RTVIObserver, RTVIProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCOutputTransport
from pydantic import BaseModel

from nvidia_pipecat.frames.riva import RivaTTSUpdateSettingsFrame, RivaVoicesFrame
from nvidia_pipecat.frames.transcripts import (
    BotUpdatedSpeakingTranscriptFrame,
    UserStoppedSpeakingTranscriptFrame,
    UserUpdatedSpeakingTranscriptFrame,
)
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.riva_speech import NemotronTTSService


class NvidiaRTVIInput(RTVIProcessor):
    """NVIDIA RTVI Input Processor for handling client messages.

    This processor extends the base RTVIProcessor to handle NVIDIA Nemotron Speech ASR and TTS Services specific
    client messages such as context resets, voice changes, and audio uploads.

    Attributes:
        _context: LLMContext for this connection.
        _upload_chunks_map: Tracks chunked uploads in progress.
        _custom_prompts_registry: Registry of custom prompts (prompt_id -> file_path).
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        **kwargs,
    ):
        """Initialize the NVIDIA RTVI input processor.

        Args:
            context: The LLM context for this connection.
            **kwargs: Additional arguments for RTVIProcessor.
        """
        super().__init__(**kwargs)

        self._context = context
        self._upload_chunks_map: dict[str, dict] = {}
        self._custom_prompts_registry: dict[str, Path] = {}

        # Register the client message handler
        self._register_event_handler("on_client_message")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle cleanup on EndFrame."""
        await super().process_frame(frame, direction)

        if isinstance(frame, EndFrame):
            await self._cleanup()

    async def _cleanup(self):
        """Clean up all state for this connection."""
        # Clean up uploaded audio prompts
        for prompt_id, prompt_path in list(self._custom_prompts_registry.items()):
            try:
                if prompt_path.exists():
                    prompt_path.unlink()
                    logger.info(f"Cleaned up custom prompt file: {prompt_path}")
            except Exception as e:
                logger.error(f"Failed to clean up prompt {prompt_id}: {e}")

        self._custom_prompts_registry.clear()
        self._upload_chunks_map.clear()
        logger.info("Cleaned up connection state")

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        """Override to handle our custom event."""
        if event_name == "on_client_message":
            await self._handle_custom_client_message(*args, **kwargs)
        else:
            await super()._call_event_handler(event_name, *args, **kwargs)

    async def _handle_custom_client_message(self, msg: RTVIClientMessage):
        """Handle custom client messages.

        Args:
            msg: The client message to handle.
        """
        msg_type = msg.type
        data = msg.data

        try:
            if msg_type == "context_reset":
                await self._handle_context_reset(msg, data)
            elif msg_type == "begin_conversation":
                await self._handle_begin_conversation(msg)
            elif msg_type == "set_tts_voice":
                await self._handle_set_tts_voice(msg, data)
            elif msg_type == "upload_custom_audio_prompt":
                await self._handle_upload_custom_audio_prompt(msg, data)
            else:
                # Unknown message type - send error response
                await self.send_error_response(msg, f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error handling client message '{msg_type}': {e}")
            import traceback

            logger.error(traceback.format_exc())
            await self.send_error_response(msg, f"Error processing message: {str(e)}")

    async def _handle_context_reset(self, msg: RTVIClientMessage, data: Any):
        """Handle context reset messages.

        Args:
            msg: The client message.
            data: List of {"role": "...", "content": "..."} message dicts.
        """
        if not isinstance(data, list):
            await self.send_error_response(msg, "Invalid prompt payload: expected list of {role, content}")
            return

        messages: list[dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in {"system", "user", "assistant"}:
                continue
            if not isinstance(content, str):
                continue
            messages.append({"role": role, "content": content})

        if not messages:
            await self.send_error_response(msg, "Invalid prompt payload")
            return

        self._context.set_messages(messages)
        logger.info(f"Context reset with {len(messages)} message(s)")
        await self.send_server_response(msg, {"status": "success"})

    async def _handle_begin_conversation(self, msg: RTVIClientMessage):
        """Handle begin conversation messages."""
        messages = self._context.get_messages().copy()
        messages.append({"role": "user", "content": "Please introduce yourself."})
        await self.push_frame(LLMMessagesUpdateFrame(messages=messages, run_llm=True))
        await self.send_server_response(msg, {"status": "started"})

    async def _handle_set_tts_voice(self, msg: RTVIClientMessage, data: Any):
        """Handle TTS voice change messages.

        Args:
            msg: The client message.
            data: Voice configuration data.
        """
        voice_type = data.get("voice_type", "").strip()
        identifier = data.get("voice_id" if voice_type == "default" else "prompt_id", "").strip()

        if not voice_type or not identifier:
            await self.send_error_response(msg, "Missing voice_type or identifier")
            return

        # For custom voices, look up the prompt path from registry
        custom_prompt_path = None
        if voice_type == "custom" and identifier != "backend":
            custom_prompt_path = self._custom_prompts_registry.get(identifier)
            if not custom_prompt_path:
                logger.warning(f"Custom prompt {identifier} not found in registry")

        # Queue frame to update TTS settings
        tts_frame = RivaTTSUpdateSettingsFrame(
            voice_type=voice_type, identifier=identifier, custom_prompt_path=custom_prompt_path
        )
        await self.push_frame(tts_frame)

        logger.info(f"TTS voice changed to {voice_type}:{identifier}")
        await self.send_server_response(msg, {"status": "updated", "voice_type": voice_type, "identifier": identifier})

    async def _handle_upload_custom_audio_prompt(self, msg: RTVIClientMessage, data: Any):
        """Handle custom audio prompt upload messages.

        Supports both direct uploads and chunked uploads for large files.

        Args:
            msg: The client message.
            data: Upload data containing audio and metadata.
        """
        # Check if this is a chunked upload
        chunk_index = data.get("chunk_index")
        total_chunks = data.get("total_chunks")
        prompt_id = data.get("prompt_id")

        audio_base64: str | None = None

        if chunk_index is not None and total_chunks is not None:
            # Handle chunked upload
            audio_base64 = await self._handle_chunked_upload(msg, data, chunk_index, total_chunks, prompt_id)
            if audio_base64 is None:
                # Still waiting for more chunks or error occurred
                return
        else:
            # Direct upload (no chunking)
            audio_base64 = data.get("audio")
            if not audio_base64:
                await self.send_error_response(msg, "No audio data provided")
                return

        # Process the complete audio
        filename = data.get("filename", "custom_voice.wav")
        await self._process_audio_upload(msg, audio_base64, filename, prompt_id)

    async def _handle_chunked_upload(
        self,
        msg: RTVIClientMessage,
        data: dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        prompt_id: str,
    ) -> str | None:
        """Handle chunked audio upload.

        Args:
            msg: The client message.
            data: Upload data.
            chunk_index: Index of this chunk.
            total_chunks: Total number of chunks.
            prompt_id: Unique prompt identifier.

        Returns:
            Complete base64 audio string if all chunks received, None otherwise.
        """
        chunk_data = data.get("chunk")
        if not chunk_data:
            await self.send_error_response(msg, f"Missing chunk data for chunk {chunk_index}")
            return None

        filename = data.get("filename", "custom_voice.wav")

        # Initialize chunk storage if needed
        if prompt_id not in self._upload_chunks_map:
            self._upload_chunks_map[prompt_id] = {
                "chunks": {},
                "total_chunks": total_chunks,
                "filename": filename,
                "prompt_id": prompt_id,
            }

        self._upload_chunks_map[prompt_id]["chunks"][chunk_index] = chunk_data

        logger.debug(f"Received chunk {chunk_index + 1}/{total_chunks} for prompt {prompt_id}")

        # Check if all chunks received
        if len(self._upload_chunks_map[prompt_id]["chunks"]) == total_chunks:
            # Reassemble chunks
            audio_base64 = "".join(self._upload_chunks_map[prompt_id]["chunks"][i] for i in range(total_chunks))
            # Clean up chunk storage
            del self._upload_chunks_map[prompt_id]
            logger.info(f"All chunks received for prompt {prompt_id}, reassembled {len(audio_base64)} bytes")
            return audio_base64
        else:
            # Still waiting for more chunks - send acknowledgment
            await self.send_server_response(
                msg,
                {
                    "status": "chunk_received",
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                },
            )
            return None

    async def _process_audio_upload(
        self,
        msg: RTVIClientMessage,
        audio_base64: str,
        filename: str,
        prompt_id: str,
    ):
        """Process and save uploaded audio.

        Args:
            msg: The client message.
            audio_base64: Base64-encoded audio data.
            filename: Original filename.
            prompt_id: Unique prompt identifier.
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"Decoded audio: {len(audio_bytes)} bytes for prompt {prompt_id}")

            # Create temp directory if it doesn't exist
            temp_dir = Path("/tmp/custom_voices")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Use the prompt_id from frontend as the filename to ensure matching
            temp_path = temp_dir / prompt_id

            # Write audio file
            temp_path.write_bytes(audio_bytes)
            logger.info(f"Saved audio to {temp_path}")

            # Register prompt in instance registry
            self._custom_prompts_registry[prompt_id] = temp_path
            logger.info(f"Registered custom prompt {prompt_id}")

            # Send success response
            await self.send_server_response(
                msg,
                {
                    "status": "uploaded",
                    "prompt_id": prompt_id,
                    "filename": filename,
                    "size": len(audio_bytes),
                },
            )

        except Exception as e:
            logger.error(f"Failed to process audio upload: {e}")
            import traceback

            logger.error(traceback.format_exc())
            await self.send_error_response(msg, f"Failed to process audio upload: {str(e)}")


class Transcript(BaseModel):
    """Transcript model for the websocket."""

    text: str
    actor: str
    message_id: str


class NvidiaRTVIObserver(RTVIObserver):
    """Forward NVIDIA Nemotron Speech ASR and TTS Services transcript and TTS config frames to an RTVI server channel.

    Aggregates incremental bot transcripts so the UI receives the full text
    so far, and assigns stable message ids per speaking turn for both user and
    bot to make UI-side synchronization straightforward. Also forwards Nemotron Speech TTS
    configuration frames for voice listing and current voice.
    """

    def __init__(self, rtvi: RTVIProcessor):
        """Initialize the NVIDIA RTVI output processor.

        Args:
            rtvi: The `RTVIProcessor` used to send server messages to the UI.
        """
        super().__init__()
        self._rtvi = rtvi
        self._message_id_user = uuid.uuid4()
        self._message_id_bot = uuid.uuid4()
        self._last_bot_transcript = ""

    async def on_push_frame(self, data: FramePushed):
        """Process the frame and send the transcript to the rtvi."""
        await super().on_push_frame(data)

        frame = data.frame
        if isinstance(frame, BotUpdatedSpeakingTranscriptFrame) and isinstance(
            data.source, BotTranscriptSynchronization
        ):
            if self._last_bot_transcript != frame.transcript:
                self._last_bot_transcript += " " + frame.transcript
            if self._rtvi is not None:
                await self._rtvi.send_server_message(
                    Transcript(
                        text=self._last_bot_transcript, actor="bot", message_id=str(self._message_id_bot)
                    ).model_dump_json()
                )
        elif isinstance(frame, BotStoppedSpeakingFrame) and isinstance(data.source, SmallWebRTCOutputTransport):
            self._message_id_bot = uuid.uuid4()
            self._last_bot_transcript = ""

        elif isinstance(frame, UserUpdatedSpeakingTranscriptFrame) and isinstance(
            data.source, UserTranscriptSynchronization
        ):
            if self._rtvi is not None:
                await self._rtvi.send_server_message(
                    Transcript(
                        text=frame.transcript, actor="user", message_id=str(self._message_id_user)
                    ).model_dump_json()
                )

        elif isinstance(frame, UserStoppedSpeakingTranscriptFrame) and isinstance(
            data.source, UserTranscriptSynchronization
        ):
            if self._rtvi is not None:
                await self._rtvi.send_server_message(
                    Transcript(
                        text=frame.transcript, actor="user", message_id=str(self._message_id_user)
                    ).model_dump_json()
                )
            self._message_id_user = uuid.uuid4()

        elif isinstance(frame, RivaVoicesFrame) and isinstance(data.source, NemotronTTSService):
            if self._rtvi is not None:
                await self._rtvi.send_server_message(
                    json.dumps(
                        {
                            "type": "riva_voices",
                            "available_voices": frame.available_voices,
                            "current_voice_id": frame.current_voice_id,
                            "is_zeroshot_model": frame.is_zeroshot_model,
                            "zero_shot_prompt": frame.zero_shot_prompt,
                        }
                    )
                )
        elif isinstance(frame, RivaTTSUpdateSettingsFrame) and isinstance(data.source, NemotronTTSService):
            if self._rtvi is not None:
                await self._rtvi.send_server_message(
                    json.dumps(
                        {
                            "type": "tts_update_settings",
                            "voice_type": frame.voice_type,
                            "voice_id": frame.identifier,
                            "language_code": frame.language_code,
                        }
                    )
                )
