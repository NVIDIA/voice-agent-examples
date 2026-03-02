# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA WebSocket Serializer Implementation.

This module defines the `NvidiaWebsocketSerializer` class, which is responsible for
serializing and deserializing frames for WebSocket communication in a speech-based
user interface. The serializer supports various frame types related to audio, text-to-speech (TTS),
and automatic speech recognition (ASR).

The serializer handles the following frame types:
- AudioRawFrame: Raw audio data
- BotUpdatedSpeakingTranscriptFrame: Updates during bot speech
- BotStoppedSpeakingFrame: End of bot speech
- UserUpdatedSpeakingTranscriptFrame: Updates during user speech
- UserStoppedSpeakingTranscriptFrame: End of user speech
- InputAudioRawFrame: Raw input audio data

The serialization format is either binary (for audio data) or JSON (for transcript updates).
"""

import io
import json
import wave

from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
)
from pipecat.serializers.base_serializer import (
    FrameSerializer,
    FrameSerializerType,
)

from nvidia_pipecat.frames.riva import RivaVoicesFrame
from nvidia_pipecat.frames.transcripts import (
    BotUpdatedSpeakingTranscriptFrame,
    UserStoppedSpeakingTranscriptFrame,
    UserUpdatedSpeakingTranscriptFrame,
)


class NvidiaWebsocketSerializer(FrameSerializer):
    """Serializes frames for WebSocket communication in NVIDIA speech interface.

    This class provides methods to serialize and deserialize frames for communication
    between the server and a speech-based UI. It supports both binary audio data
    and JSON-formatted transcript updates.

    Attributes:
        type (FrameSerializerType): The serializer type, always BINARY.

    Input Frames:
        AudioRawFrame: Raw audio data
        BotUpdatedSpeakingTranscriptFrame: TTS update
        BotStoppedSpeakingFrame: TTS end
        UserUpdatedSpeakingTranscriptFrame: ASR update
        UserStoppedSpeakingTranscriptFrame: ASR end
        InputAudioRawFrame: Raw input audio
    """

    @property
    def type(self) -> FrameSerializerType:
        """Return the type of FrameSerializer.

        Returns:
            FrameSerializerType: Always returns BINARY type for this serializer.
        """
        return FrameSerializerType.BINARY

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a frame to JSON string or bytes.

        Args:
            frame (Frame): The frame to serialize. Can be one of:
                - AudioRawFrame: Returns raw audio bytes
                - BotUpdatedSpeakingTranscriptFrame: Returns JSON with TTS update
                - BotStoppedSpeakingFrame: Returns JSON with TTS end
                - UserUpdatedSpeakingTranscriptFrame: Returns JSON with ASR update
                - UserStoppedSpeakingTranscriptFrame: Returns JSON with ASR end

        Returns:
            str | bytes | None: Serialized data:
                - bytes for audio frames
                - JSON string for transcript updates
                - None for unsupported frames
        """
        if isinstance(frame, AudioRawFrame):
            return frame.audio

        message = None
        if isinstance(frame, BotUpdatedSpeakingTranscriptFrame):
            message = {"type": "tts_update", "tts": frame.transcript}
        elif isinstance(frame, BotStoppedSpeakingFrame):
            message = {"type": "tts_end"}
        elif isinstance(frame, UserUpdatedSpeakingTranscriptFrame):
            message = {"type": "asr_update", "asr": frame.transcript}
        elif isinstance(frame, UserStoppedSpeakingTranscriptFrame):
            message = {"type": "asr_end", "asr": frame.transcript}
        elif isinstance(frame, RivaVoicesFrame):
            message = {
                "type": "riva_voices",
                "available_voices": frame.available_voices,
                "current_voice_id": frame.current_voice_id,
                "is_zeroshot_model": frame.is_zeroshot_model,
                "zero_shot_prompt": frame.zero_shot_prompt,
            }

        if message:
            return json.dumps(message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize bytes into a Frame object.

        Args:
            data (str | bytes): The data to deserialize. Expected to be
                WAV-formatted audio data.

        Returns:
            Frame | None: The deserialized frame as an InputAudioRawFrame for audio data,
                or None for unsupported data types.
        """
        if not isinstance(data, bytes):
            return None

        try:
            with io.BytesIO(data) as buffer, wave.open(buffer, "rb") as wf:
                return InputAudioRawFrame(
                    wf.readframes(wf.getnframes()),
                    wf.getframerate(),
                    wf.getnchannels(),
                )
        except (wave.Error, EOFError) as e:
            logger.error(f"Failed to deserialize audio data: {e}")
            return None
