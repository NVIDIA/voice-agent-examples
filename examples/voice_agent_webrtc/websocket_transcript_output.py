# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Websocket transcript output for the voice agent webrtc demo."""

import uuid

from fastapi import WebSocket
from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pydantic import BaseModel

from nvidia_pipecat.frames.transcripts import (
    BotUpdatedSpeakingTranscriptFrame,
    UserStoppedSpeakingTranscriptFrame,
    UserUpdatedSpeakingTranscriptFrame,
)


class Transcript(BaseModel):
    """Transcript model for the websocket."""

    text: str
    actor: str
    message_id: str


class WebsocketTranscriptOutput(FrameProcessor):
    """Frame processor to send transcripts to the websocket."""

    def __init__(self, ws: WebSocket, name: str | None = None, metrics: FrameProcessorMetrics | None = None, **kwargs):
        """Initialize the frame processor.

        Args:
            ws (WebSocket): The websocket to send the transcripts to.
            name (str): The name of the frame processor.
            metrics (FrameProcessorMetrics): The metrics for the frame processor.
            kwargs (dict): Additional keyword arguments.
        """
        super().__init__(name=name, metrics=metrics, **kwargs)
        self._ws = ws
        self._message_id_user = uuid.uuid4()
        self._message_id_bot = uuid.uuid4()
        self._last_bot_transcript = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process the frame and send the transcript to the websocket."""
        await super().process_frame(frame, direction)
        if isinstance(frame, BotUpdatedSpeakingTranscriptFrame):
            if self._last_bot_transcript != frame.transcript:
                self._last_bot_transcript += " " + frame.transcript
            if self._ws is not None:
                await self._ws.send_text(
                    Transcript(
                        text=self._last_bot_transcript, actor="bot", message_id=str(self._message_id_bot)
                    ).model_dump_json()
                )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._message_id_bot = uuid.uuid4()
            self._last_bot_transcript = ""
        elif isinstance(frame, UserUpdatedSpeakingTranscriptFrame):
            if self._ws is not None:
                await self._ws.send_text(
                    Transcript(
                        text=frame.transcript, actor="user", message_id=str(self._message_id_user)
                    ).model_dump_json()
                )
        elif isinstance(frame, UserStoppedSpeakingTranscriptFrame):
            if self._ws is not None:
                await self._ws.send_text(
                    Transcript(
                        text=frame.transcript, actor="user", message_id=str(self._message_id_user)
                    ).model_dump_json()
                )
            self._message_id_user = uuid.uuid4()
        await super().push_frame(frame, direction)
