# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the Nvidia TTS Response Cacher."""

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test as run_pipecat_test

from nvidia_pipecat.processors.nvidia_context_aggregator import NvidiaTTSResponseCacher


@pytest.mark.asyncio()
async def test_nvidia_tts_response_cacher():
    """Tests NvidiaTTSResponseCacher's response deduplication functionality.

    Tests the cacher's ability to deduplicate TTS audio responses in a sequence
    of frames including user speech events and LLM responses.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Correct handling of user speech start/stop frames
        - Deduplication of identical TTS audio frames
        - Preservation of LLM response start/end frames
        - Frame ordering in pipeline output
        - Only one TTS frame is retained
    """
    nvidia_tts_response_cacher = NvidiaTTSResponseCacher()
    pipeline = Pipeline([nvidia_tts_response_cacher])

    test_audio = b"\x52\x49\x46\x46\x24\x08\x00\x00\x57\x41\x56\x45\x66\x6d\x74\x20"
    frames_to_send = [
        UserStartedSpeakingFrame(),
        LLMFullResponseStartFrame(),
        TTSAudioRawFrame(
            audio=test_audio,
            sample_rate=16000,
            num_channels=1,
        ),
        LLMFullResponseEndFrame(),
        LLMFullResponseStartFrame(),
        TTSAudioRawFrame(
            audio=test_audio,
            sample_rate=16000,
            num_channels=1,
        ),
        LLMFullResponseEndFrame(),
        UserStoppedSpeakingFrame(),
    ]

    expected_down_frames = [
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        LLMFullResponseStartFrame,
        TTSAudioRawFrame,
        LLMFullResponseEndFrame,
    ]

    received_down_frames, _ = await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify we only got one TTS frame
    tts_frames = [f for f in received_down_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(tts_frames) == 1
