# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Utility functions and classes for testing pipelines and processors.

This module provides various test utilities including frame processors, test runners,
and helper functions for frame comparison in tests.
"""

import asyncio
from asyncio.tasks import sleep
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import ANY

import numpy as np
from pipecat.frames.frames import EndFrame, Frame, InputAudioRawFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor, SleepFrame
from pipecat.tests.utils import run_test as run_pipecat_test

from nvidia_pipecat.frames.action import ActionFrame


def ignore(frame: Frame, *args: str):
    """Return a copy of the frames with attributes listed in *args set to unittest.mock.ANY.

    Any attribute listed in args will be ignored when using comparisons against the returned frame.

    Args:
        frame (Frame): Frame to create a copy from and set selected attributes to unittest.mock.ANY.
        *args (str): Attribute names. Special values to ignore common sets of attributes:
            'ids': ignore standard frame attributes related to frame ids
            'all_ids': ignore standard frame attributes related to frame and action ids
            'timestamps': ignore standard frame attributes containing timestamps

    Returns:
        Frame: A copy of the input frame with specified attributes set to ANY.

    Raises:
        ValueError: If an attribute name is not found in the frame.
    """
    new_frame = copy(frame)
    for arg in args:
        is_updated = False
        if arg == "all_ids" or arg == "ids":
            new_frame.id = ANY
            new_frame.name = ANY
            is_updated = True
        if arg == "all_ids":
            new_frame.action_id = ANY
            is_updated = True
        if arg == "timestamps":
            new_frame.pts = ANY
            new_frame.action_started_at = ANY
            new_frame.action_finished_at = ANY
            new_frame.action_updated_at = ANY
            is_updated = True

        if hasattr(new_frame, arg):
            setattr(new_frame, arg, ANY)
        elif not is_updated:
            raise ValueError(
                f"Frame {frame.__class__.__name__} has not attribute '{arg}' to ignore. Did you misspell it?"
            )
    return new_frame


def ignore_ids(frame: Frame) -> Frame:
    """Return a copy of the frame that matches any id and name.

    This is useful if you do not want to assert a specific ID and name.

    Args:
        frame (Frame): The frame to create a copy from.

    Returns:
        Frame: A copy of the input frame with ID and name set to ANY.
    """
    new_frame = copy(frame)
    new_frame.id = ANY
    new_frame.name = ANY
    # Volatile attributes that vary by runtime context and should not
    # participate in equality assertions within tests
    if hasattr(new_frame, "processor"):
        new_frame.processor = ANY
    if hasattr(new_frame, "transport_source"):
        new_frame.transport_source = ANY
    if hasattr(new_frame, "transport_destination"):
        new_frame.transport_destination = ANY
    if isinstance(frame, ActionFrame):
        new_frame.action_id = ANY
    return new_frame


def ignore_timestamps(frame: Frame) -> Frame:
    """Return a copy of the frame that matches frames ignoring any timestamps.

    Args:
        frame (Frame): The frame to create a copy from.

    Returns:
        Frame: A copy of the input frame with all timestamp fields set to ANY.
    """
    new_frame = copy(frame)
    new_frame.pts = ANY
    new_frame.action_started_at = ANY
    new_frame.action_finished_at = ANY
    new_frame.action_updated_at = ANY
    return new_frame


@dataclass
class FrameHistoryEntry:
    """Storing a frame and the frame direction.

    Attributes:
        frame (Frame): The stored frame.
        direction (FrameDirection): The direction of the frame in the pipeline.
    """

    frame: Frame
    direction: FrameDirection


class FrameStorage(FrameProcessor):
    """A frame processor that stores all received frames in memory for inspection.

    This processor maintains a history of all frames that pass through it, along with their
    direction, allowing for later inspection and verification in tests.

    Attributes:
        history (list[FrameHistoryEntry]): List of all frames that have passed through the processor.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the FrameStorage processor.

        Args:
            **kwargs: Additional keyword arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.history: list[FrameHistoryEntry] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame by storing it in history and forwarding it.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction the frame is traveling in the pipeline.
        """
        await super().process_frame(frame, direction)

        self.history.append(FrameHistoryEntry(frame, direction))
        await self.push_frame(frame, direction)

    def frames_of_type(self, t) -> list[Frame]:
        """Get all frames of a specific type from the history.

        Args:
            t: The type of frames to filter for.

        Returns:
            list[Frame]: List of all frames matching the specified type.
        """
        return [e.frame for e in self.history if isinstance(e.frame, t)]

    async def wait_for_frame(self, frame: Frame, timeout: timedelta = timedelta(seconds=5.0)) -> None:
        """Block until a matching frame is found in history.

        Args:
            frame (Frame): The frame to wait for.
            timeout (timedelta, optional): Maximum time to wait. Defaults to 5 seconds.

        Raises:
            TimeoutError: If the frame is not found within the timeout period.
        """
        candidates = {}

        def is_same_frame(frame_a, frame_b) -> bool:
            if type(frame_a) is not type(frame_b):
                return False

            if frame_a == frame_b:
                return True
            else:
                candidates[frame_a.id] = frame_a.__repr__()
                return False

        found = False
        start_time = datetime.now()
        while not found:
            found = any([is_same_frame(entry.frame, frame) for entry in self.history])
            if not found:
                if datetime.now() - start_time > timeout:
                    raise TimeoutError(
                        "Frame not found until timeout reached.\n"
                        f"EXPECTED:\n{frame.__repr__()}\n"
                        f"FOUND\n{'\n'.join(candidates.values())}"
                    )
                await asyncio.sleep(0.01)


class SinusWaveProcessor(FrameProcessor):
    """A frame processor that generates a sine wave audio signal.

    This processor generates a continuous sine wave at 440 Hz (A4 note) when started,
    and outputs it as audio frames. It is useful for testing audio processing pipelines.

    Attributes:
        duration (timedelta): The total duration of the sine wave to generate.
        chunk_duration (float): Duration of each audio chunk in seconds.
        audio_frame_count (int): Total number of audio frames to generate.
        audio_task (asyncio.Task | None): Task handling the audio generation.
    """

    def __init__(
        self,
        *,
        duration: timedelta,
        **kwargs,
    ):
        """Initialize the SinusWaveProcessor.

        Args:
            duration (timedelta): The total duration of the sine wave to generate.
            **kwargs: Additional keyword arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.duration = duration
        self.audio_task: asyncio.Task | None = None

        self.chunk_duration = 0.02  # 20 milliseconds
        self.audio_frame_count = round(self.duration.total_seconds() / self.chunk_duration)

    async def _cancel_audio_task(self):
        """Cancel the running audio generation task if it exists."""
        if self.audio_task and not self.audio_task.done():
            await self.cancel_task(self.audio_task)
            self.audio_task = None

    async def stop(self):
        """Stop the audio generation by canceling the audio task."""
        await self._cancel_audio_task()

    async def cleanup(self):
        """Clean up resources by stopping audio generation and calling parent cleanup."""
        await super().cleanup()
        await self._cancel_audio_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames to start/stop audio generation.

        Starts audio generation on StartFrame and stops on EndFrame.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction the frame is traveling in the pipeline.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.audio_task = self.create_task(self.start())
        if isinstance(frame, EndFrame):
            await self.stop()
        await super().push_frame(frame, direction)

    async def start(self):
        """Start generating and outputting sine wave audio frames.

        Generates a continuous 440 Hz sine wave, split into 20ms chunks,
        and outputs them as InputAudioRawFrame instances.
        """
        sample_rate = 16000  # Hz
        frequency = 440  # Hz (A4 tone)

        phase_offset = 0
        for _ in range(self.audio_frame_count):
            chunk_samples = int(sample_rate * self.chunk_duration)

            # Generate the time axis
            t = np.arange(chunk_samples) / sample_rate

            # Create the sine wave with the given phase offset
            sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t + phase_offset)

            # Calculate the new phase offset for the next chunk
            phase_offset = (2 * np.pi * frequency * self.chunk_duration + phase_offset) % (2 * np.pi)

            sine_wave_pcm = (sine_wave * 32767).astype(np.int16)

            pipecat_frame = InputAudioRawFrame(audio=sine_wave_pcm.tobytes(), sample_rate=sample_rate, num_channels=1)
            await super().push_frame(pipecat_frame)

            await sleep(self.chunk_duration)


async def run_test(
    processor: FrameProcessor,
    *,
    frames_to_send: Sequence[Frame],
    expected_down_frames: Sequence[Frame],
    expected_up_frames: Sequence[Frame] = [],
    ignore_start: bool = True,
    start_metadata: dict[str, Any] | None = None,
    send_end_frame: bool = True,
) -> tuple[Sequence[Frame], Sequence[Frame]]:
    """Run a test on a frame processor with predefined input and expected output frames.

    Args:
        processor (FrameProcessor): The processor to test.
        frames_to_send (Sequence[Frame]): Frames to send through the processor.
        expected_down_frames (Sequence[Frame]): Expected frames in downstream direction.
        expected_up_frames (Sequence[Frame], optional): Expected frames in upstream direction.
            Defaults to [].
        ignore_start (bool, optional): Whether to ignore start frames. Defaults to True.
        start_metadata (dict[str, Any], optional): Metadata to include in start frame.
            Defaults to None.
        send_end_frame (bool, optional): Whether to send an end frame. Defaults to True.

    Returns:
        tuple[Sequence[Frame], Sequence[Frame]]: Tuple of (received downstream frames, received upstream frames).

    Raises:
        AssertionError: If received frames don't match expected frames.
    """
    # If start_metadata is provided, use the interactive runner so we can set PipelineParams(start_metadata=...)
    if start_metadata:

        async def scenario(task: PipelineTask):
            for f in frames_to_send:
                # Emulate timing semantics of SleepFrame instead of pushing it downstream
                if isinstance(f, SleepFrame):
                    await asyncio.sleep(getattr(f, "sleep", 0))
                else:
                    await task.queue_frame(f)

        received_down_frames, received_up_frames = await run_interactive_test(
            processor,
            test_coroutine=scenario,
            start_metadata=start_metadata,
            ignore_start=ignore_start,
            send_end_frame=send_end_frame,
        )
    else:
        received_down_frames, received_up_frames = await run_pipecat_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[f.__class__ for f in expected_down_frames],
            expected_up_frames=[f.__class__ for f in expected_up_frames],
            ignore_start=ignore_start,
            send_end_frame=send_end_frame,
        )

    # Filter out control/sleep frames to mirror upstream test helper behavior
    received_down_frames = [f for f in received_down_frames if not isinstance(f, SleepFrame)]

    for real, expected in zip(received_up_frames, expected_up_frames, strict=True):
        assert real == expected, f"Frame mismatch: \nreal: {repr(real)} \nexpected: {repr(expected)}"

    for real, expected in zip(received_down_frames, expected_down_frames, strict=True):
        assert real == expected, f"Frame mismatch: \nreal: {repr(real)} \nexpected: {repr(expected)}"

    return received_down_frames, received_up_frames


async def run_interactive_test(
    processor: FrameProcessor,
    *,
    test_coroutine,
    start_metadata: dict[str, Any] | None = None,
    ignore_start: bool = True,
    send_end_frame: bool = True,
) -> tuple[Sequence[Frame], Sequence[Frame]]:
    """Run an interactive test on a frame processor with a custom test coroutine.

    This function allows for more complex testing scenarios where frames need to be
    sent and received dynamically during the test.

    Args:
        processor (FrameProcessor): The processor to test.
        test_coroutine: Coroutine function that implements the test logic.
        start_metadata (dict[str, Any], optional): Metadata to include in start frame.
            Defaults to None.
        ignore_start (bool, optional): Whether to ignore start frames. Defaults to True.
        send_end_frame (bool, optional): Whether to send an end frame. Defaults to True.

    Returns:
        tuple[Sequence[Frame], Sequence[Frame]]: Tuple of (received downstream frames, received upstream frames).
    """
    if start_metadata is None:
        start_metadata = {}
    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up,
        queue_direction=FrameDirection.UPSTREAM,
        ignore_start=ignore_start,
    )
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=ignore_start,
    )

    pipeline = Pipeline([source, processor, sink])

    task = PipelineTask(pipeline, params=PipelineParams(start_metadata=start_metadata, allow_interruptions=False))

    async def run_test():
        # Just give a little head start to the runner.
        await asyncio.sleep(0.01)
        await test_coroutine(task)

        if send_end_frame:
            await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), run_test())

    #
    # Down frames
    #
    received_down_frames: list[Frame] = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame) or not send_end_frame:
            received_down_frames.append(frame)

    #
    # Up frames
    #
    received_up_frames: list[Frame] = []
    while not received_up.empty():
        frame = await received_up.get()
        received_up_frames.append(frame)

    return (received_down_frames, received_up_frames)
