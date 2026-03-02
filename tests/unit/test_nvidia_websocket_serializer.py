# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the NvidiaWebsocketSerializer class.

This module contains unit tests for the `NvidiaWebsocketSerializer` class, which is responsible
for serializing and deserializing frames for WebSocket communication in a speech-based user interface.
The tests cover various frame types related to audio, text-to-speech (TTS), and automatic speech recognition (ASR).

The test suite verifies:
1. Serialization of bot speech updates and end events
2. Serialization of user speech updates and end events
3. Serialization of raw audio frames
4. Handling of unsupported frame types

Each test case validates the correct format and content of the serialized data,
ensuring proper JSON formatting for transcript updates and binary data handling for audio frames.
"""

import json

import pytest
from pipecat.frames.frames import AudioRawFrame, BotStoppedSpeakingFrame, Frame

from nvidia_pipecat.frames.transcripts import (
    BotUpdatedSpeakingTranscriptFrame,
    UserStoppedSpeakingTranscriptFrame,
    UserUpdatedSpeakingTranscriptFrame,
)
from nvidia_pipecat.serializers.nvidia_websocket_serializer import NvidiaWebsocketSerializer


@pytest.fixture
def serializer():
    """Fixture to create an instance of NvidiaWebsocketSerializer.

    Returns:
        NvidiaWebsocketSerializer: A fresh instance of the serializer for each test.
    """
    return NvidiaWebsocketSerializer()


@pytest.mark.asyncio
async def test_serialize_bot_updated_speaking_frame(serializer):
    """Test serialization of BotUpdatedSpeakingTranscriptFrame.

    This test verifies that when a bot speech update frame is serialized,
    it produces the correct JSON format with 'tts_update' type and the transcript.

    Args:
        serializer: The NvidiaWebsocketSerializer fixture.
    """
    frame = BotUpdatedSpeakingTranscriptFrame(transcript="test_transcript")
    result = await serializer.serialize(frame)
    expected_result = json.dumps({"type": "tts_update", "tts": "test_transcript"})
    assert result == expected_result


@pytest.mark.asyncio
async def test_serialize_bot_stopped_speaking_frame(serializer):
    """Test serialization of BotStoppedSpeakingFrame.

    This test verifies that when a bot speech end frame is serialized,
    it produces the correct JSON format with 'tts_end' type.

    Args:
        serializer: The NvidiaWebsocketSerializer fixture.
    """
    frame = BotStoppedSpeakingFrame()
    result = await serializer.serialize(frame)
    expected_result = json.dumps({"type": "tts_end"})
    assert result == expected_result


@pytest.mark.asyncio
async def test_serialize_user_started_speaking_frame(serializer):
    """Test serialization of UserUpdatedSpeakingTranscriptFrame.

    This test verifies that when a user speech update frame is serialized,
    it produces the correct JSON format with 'asr_update' type and the transcript.

    Args:
        serializer: The NvidiaWebsocketSerializer fixture.
    """
    frame = UserUpdatedSpeakingTranscriptFrame(transcript="test_transcript")
    result = await serializer.serialize(frame)
    expected_result = json.dumps({"type": "asr_update", "asr": "test_transcript"})
    assert result == expected_result


@pytest.mark.asyncio
async def test_serialize_user_stopped_speaking_frame(serializer):
    """Test serialization of UserStoppedSpeakingTranscriptFrame.

    This test verifies that when a user speech end frame is serialized,
    it produces the correct JSON format with 'asr_end' type and the transcript.

    Args:
        serializer: The NvidiaWebsocketSerializer fixture.
    """
    frame = UserStoppedSpeakingTranscriptFrame(transcript="test_asr_transcript")
    result = await serializer.serialize(frame)
    expected_result = json.dumps({"type": "asr_end", "asr": "test_asr_transcript"})
    assert result == expected_result


@pytest.mark.asyncio
async def test_serialize_audio_raw_frame(serializer):
    """Test serialization of AudioRawFrame.

    This test verifies that when an audio frame is serialized,
    it returns the raw audio bytes without any modification.

    Args:
        serializer: The NvidiaWebsocketSerializer fixture.
    """
    frame = AudioRawFrame(audio=b"\xa2", sample_rate=16000, num_channels=1)
    result = await serializer.serialize(frame)
    expected_result = frame.audio
    assert result == expected_result


@pytest.mark.asyncio
async def test_serialize_none(serializer):
    """Test serialization of an unsupported frame type.

    This test verifies that when an unsupported frame type is serialized,
    the serializer returns None instead of raising an error.

    Args:
        serializer: The NvidiaWebsocketSerializer fixture.
    """
    frame = Frame()
    result = await serializer.serialize(frame)
    assert result is None


@pytest.mark.asyncio
async def test_deserialize_input_audio_raw_frame(serializer):
    """Test deserialization of an audio message into InputAudioRawFrame."""
    data = (
        b"RIFF$\x04\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
        + b"data\x00\x04\x00\x00"
    )
    result = await serializer.deserialize(data)
    assert result.sample_rate == 16000
    assert result.num_channels == 1
