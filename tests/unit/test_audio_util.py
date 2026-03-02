# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for audio utility functionality.

This module contains tests for audio processing components, particularly the AudioRecorder.
It verifies the basic functionality of audio recording and processing pipelines.
"""

from datetime import timedelta
from pathlib import Path

import pytest
from pipecat.frames.frames import InputAudioRawFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame

from nvidia_pipecat.processors.audio_util import AudioRecorder
from nvidia_pipecat.utils.logging import setup_default_logging
from tests.unit.utils import SinusWaveProcessor, ignore, ignore_ids, run_test


@pytest.mark.asyncio()
async def test_audio_recorder():
    """Test the AudioRecorder processor functionality.

    Tests:
        - Audio frame processing from sine wave generator
        - WAV file writing with sample rate from frame
        - Non-audio frame passthrough

    Raises:
        AssertionError: If audio file is not created or frame processing fails.
    """
    setup_default_logging(level="TRACE")

    # Delete tmp audio file if it exists
    TMP_FILE = Path("./tmp_file.wav")
    if TMP_FILE.exists():
        TMP_FILE.unlink()

    recorder = AudioRecorder(output_file=str(TMP_FILE), frame_type=InputAudioRawFrame)
    sinus = SinusWaveProcessor(duration=timedelta(seconds=0.3))

    frames_to_send = [
        SleepFrame(0.5),
        TextFrame("go through"),
    ]
    expected_down_frames = [
        ignore(InputAudioRawFrame(audio=b"1" * 640, sample_rate=16000, num_channels=1), "audio", "id", "name")
    ] * sinus.audio_frame_count + [ignore_ids(TextFrame("go through"))]
    pipeline = Pipeline([sinus, recorder])
    print(f"expected number of frames: {sinus.audio_frame_count}")
    await run_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
        start_metadata={"stream_id": "1235"},
    )

    # Make sure that audio dump was generated
    assert TMP_FILE.is_file()
