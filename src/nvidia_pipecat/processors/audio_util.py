# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Audio utilities."""

import os
import wave
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# ruff: noqa: SIM115


class AudioRecorder(FrameProcessor):
    """Records audio frames to a file.

    The writer is lazily initialized on the first matching frame, extracting
    sample rate and channel count directly from the frame.

    Args:
        output_file (str): The path to the output file.
        frame_type (type[Frame]): The type of frame to record. Defaults to AudioRawFrame.
        **kwargs: Additional keyword arguments passed to the parent FrameProcessor.
    """

    # Pipecat uses 16-bit PCM audio throughout (hardcoded in AudioRawFrame.num_frames calculation)
    BYTES_PER_SAMPLE = 2

    def __init__(
        self,
        output_file: str,
        frame_type: type[Frame] = AudioRawFrame,
        **kwargs,
    ):
        """Initialize the AudioRecorder.

        Args:
            output_file (str): Path to the output WAV file.
            frame_type (type[Frame]): Type of frame to record. Defaults to AudioRawFrame.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._output_file = output_file
        self._frame_type = frame_type
        self._writer: wave.Wave_write | None = None

    def _init_writer(self, frame: Frame):
        """Initialize the wave writer using properties from the first frame.

        Args:
            frame: The first audio frame, used to extract sample rate and channels.
        """
        Path(os.path.dirname(self._output_file)).mkdir(parents=True, exist_ok=True)
        self._writer = wave.open(self._output_file, "wb")

        sample_rate = getattr(frame, "sample_rate", 16000)
        num_channels = getattr(frame, "num_channels", 1)

        self._writer.setnchannels(num_channels)
        self._writer.setsampwidth(self.BYTES_PER_SAMPLE)
        self._writer.setframerate(sample_rate)

        logger.debug(f"AudioRecorder initialized: {self._output_file} (rate={sample_rate}, channels={num_channels})")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, self._frame_type):
            if self._writer is None:
                self._init_writer(frame)

            logger.trace(f"AudioRecorder writing frame (length: {len(frame.audio)})")
            self._writer.writeframes(frame.audio)

        await super().push_frame(frame, direction)

    async def cleanup(self):
        """Clean up the audio recorder.

        Closes the audio file writer and performs necessary cleanup operations.
        """
        await super().cleanup()
        if self._writer:
            logger.info(f"Finalizing audio file: {self._output_file}")
            self._writer.close()
            self._writer = None
