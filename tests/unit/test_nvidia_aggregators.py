# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the Nvidia Aggregators."""

import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesUpdateFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as run_pipecat_test
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.frames.riva import RivaInterimTranscriptionFrame
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaUserContextAggregator,
    create_nvidia_context_aggregator,
)


@pytest.mark.asyncio()
async def test_normal_flow():
    """Test the normal flow of user and assistant interactions with interim transcriptions enabled.

    Tests the sequence of events from user speech start through assistant response,
    verifying proper handling of interim and final transcriptions.

    The test verifies:
        - User speech start frame handling
        - Interim transcription processing
        - Final transcription handling
        - User speech stop frame handling
        - Assistant response processing
        - Context updates at each stage
    """
    messages = []
    context = LLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})
    # Test Case 1: Normal flow with UserStartedSpeakingFrame first
    frames_to_send = [
        UserStartedSpeakingFrame(),
        LLMMessagesUpdateFrame(messages=messages, run_llm=True),
        RivaInterimTranscriptionFrame("Hello", "", time_now_iso8601(), None, stability=1.0),
        TranscriptionFrame("Hello User Aggregator!", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
    ]
    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )
    expected_down_frames = [
        UserStartedSpeakingFrame,
        InterruptionFrame,
        InterruptionFrame,
        OpenAILLMContextFrame,  # From initial LLMMessagesUpdateFrame(run_llm=True) - base aggregator
        LLMContextFrame,  # From first interim - our custom aggregator
        LLMContextFrame,  # From final transcription - our custom aggregator
        UserStoppedSpeakingFrame,
        LLMContextFrame,  # From assistant response
    ]
    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "Hello User Aggregator!"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_user_speaking_frame_delay_cases():
    """Test handling of transcription frames that arrive before UserStartedSpeakingFrame.

    Tests edge cases around transcription frame timing relative to the
    UserStartedSpeakingFrame.

    The test verifies:
        - Interim frames before UserStartedSpeakingFrame are ignored
        - Low stability interim frames are ignored
        - Only processes transcriptions after UserStartedSpeakingFrame
        - Context is updated correctly for valid frames
    """
    messages = []
    context = LLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})

    # Test Case 2: RivaInterimTranscriptionFrames before UserStartedSpeakingFrame
    frames_to_send = [
        RivaInterimTranscriptionFrame(
            "Testing", "", time_now_iso8601(), None, stability=0.5
        ),  # Should be ignored (low stability)
        RivaInterimTranscriptionFrame(
            "Testing delayed", "", time_now_iso8601(), None, stability=1.0
        ),  # Should be ignored (no UserStartedSpeakingFrame yet)
        SleepFrame(0.5),
        UserStartedSpeakingFrame(),
        RivaInterimTranscriptionFrame(
            "Testing after start", "", time_now_iso8601(), None, stability=1.0
        ),  # Should be processed
        TranscriptionFrame("Testing after start complete", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
    ]

    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )
    expected_down_frames = [
        UserStartedSpeakingFrame,
        InterruptionFrame,
        InterruptionFrame,
        LLMContextFrame,  # from first interim after UserStartedSpeakingFrame
        LLMContextFrame,  # from final transcription
        UserStoppedSpeakingFrame,
        LLMContextFrame,  # From assistant response
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Testing after start complete"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_multiple_interims_with_final_transcription():
    """Test handling of multiple interim transcription frames followed by a final transcription.

    Tests the processing of a sequence of interim transcriptions followed by
    a final transcription.

    The test verifies:
        - Multiple interim transcriptions are processed correctly
        - Final transcription properly overwrites previous interims
        - Context updates occur for each valid frame
        - Message history maintains correct order
    """
    messages = []
    context = LLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})

    # Test Case 3: Multiple interim frames with final transcription
    frames_to_send = [
        UserStartedSpeakingFrame(),
        RivaInterimTranscriptionFrame("Hello", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello Again", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello Again User", "", time_now_iso8601(), None, stability=1.0),
        TranscriptionFrame("Hello Again User Aggregator!", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
    ]

    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )
    expected_down_frames = [
        UserStartedSpeakingFrame,
        InterruptionFrame,
        InterruptionFrame,
        InterruptionFrame,
        InterruptionFrame,
        LLMContextFrame,  # From interim 1
        LLMContextFrame,  # From interim 2
        LLMContextFrame,  # From interim 3
        LLMContextFrame,  # From final transcription
        UserStoppedSpeakingFrame,
        LLMContextFrame,  # From assistant response
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Hello Again User Aggregator!"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_transcription_after_user_stopped_speaking():
    """Tests handling of late transcription frames.

    Tests behavior when transcription frames arrive after UserStoppedSpeakingFrame.

    The test verifies:
        - Late transcriptions are still processed
        - Context is updated with final transcription
        - Assistant responses are handled correctly
        - Message history maintains proper sequence
    """
    messages = []
    context = LLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})

    # Test Case 4: TranscriptionFrame after UserStoppedSpeakingFrame
    frames_to_send = [
        UserStartedSpeakingFrame(),
        RivaInterimTranscriptionFrame("Late", "", time_now_iso8601(), None, stability=1.0),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
        TranscriptionFrame("Late transcription!", 1, 2),
        SleepFrame(0.1),
    ]

    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )

    expected_down_frames = [
        UserStartedSpeakingFrame,
        InterruptionFrame,
        LLMContextFrame,  # From first interim
        UserStoppedSpeakingFrame,
        InterruptionFrame,
        LLMContextFrame,  # From final after UserStoppedSpeakingFrame
        LLMContextFrame,  # From assistant response
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Late transcription!"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_no_interim_frames():
    """Tests behavior when interim frames are disabled.

    Tests the aggregator's handling of transcriptions when send_interims=False.

    The test verifies:
        - Interim frames are ignored
        - Only final transcription is processed
        - System prompts are preserved
        - Context updates occur only for final transcription
        - Assistant responses are processed correctly
    """
    messages = [{"role": "system", "content": "This is system prompt"}]
    context = LLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=False)
    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])

    frames_to_send = [
        UserStartedSpeakingFrame(),
        LLMMessagesUpdateFrame(messages=messages, run_llm=True),
        # These interim frames should be ignored due to send_interims=False
        RivaInterimTranscriptionFrame("Hello", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello there", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello there user", "", time_now_iso8601(), None, stability=1.0),
        # Only the final transcription should be processed
        TranscriptionFrame("Hello there user final!", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
        # Assistant response
        LLMFullResponseStartFrame(),
        TextFrame("Hello from assistant!"),
        LLMFullResponseEndFrame(),
    ]

    expected_down_frames = [
        UserStartedSpeakingFrame,
        InterruptionFrame,
        OpenAILLMContextFrame,  # From initial LLMMessagesUpdateFrame(run_llm=True) - base aggregator
        LLMContextFrame,  # Only from final transcription - our custom aggregator
        UserStoppedSpeakingFrame,
        LLMContextFrame,  # From assistant response
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "Hello there user final!"},
        {"role": "assistant", "content": "Hello from assistant!"},
    ]


@pytest.mark.asyncio()
async def test_get_truncated_context():
    """Tests context truncation functionality.

    Tests the get_truncated_context() method of NvidiaUserContextAggregator
    with a specified chat history limit.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Context is truncated to specified limit
        - System prompt is preserved
        - Most recent messages are retained
        - Message order is maintained
    """
    messages = [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "Hi, there!"},
        {"role": "assistant", "content": "Hello, how may I assist you?"},
        {"role": "user", "content": "How to be more productive?"},
        {"role": "assistant", "content": "Priotize the tasks, make a list..."},
        {"role": "user", "content": "What is metaverse?"},
        {
            "role": "assistant",
            "content": "The metaverse is envisioned as a digital ecosystem built on virtual 3D technology",
        },
        {
            "role": "assistant",
            "content": "It leverages 3D technology and digital"
            "representation for creating virtual environments and user experiences",
        },
        {"role": "user", "content": "thanks, Bye!"},
    ]
    context = LLMContext(messages)
    user = NvidiaUserContextAggregator(context=context, chat_history_limit=2)
    truncated_context = await user.get_truncated_context()
    assert truncated_context.get_messages() == [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "What is metaverse?"},
        {
            "role": "assistant",
            "content": "The metaverse is envisioned as a digital ecosystem built on virtual 3D technology",
        },
        {
            "role": "assistant",
            "content": "It leverages 3D technology and digital"
            "representation for creating virtual environments and user experiences",
        },
        {"role": "user", "content": "thanks, Bye!"},
    ]


@pytest.mark.asyncio()
async def test_get_truncated_context_preserve_prompt_messages():
    """Tests context truncation with preserve_prompt_messages functionality.

    Tests the get_truncated_context() method of NvidiaUserContextAggregator
    with different values of preserve_prompt_messages parameter, specifically
    testing the code that separates initial prompt messages from the rest
    (lines 258-265 in nvidia_context_aggregator.py).

    The test verifies:
        - preserve_prompt_messages=0: No initial messages preserved via that mechanism
          (note: system messages are always preserved regardless of limit)
        - preserve_prompt_messages=1: First message preserved (default)
        - preserve_prompt_messages=2: First two messages preserved (Nemotron models)
        - Initial messages are correctly separated from remaining messages
        - Truncation still works correctly on remaining messages
    """
    # Test case 1: preserve_prompt_messages=0 (no initial preservation)
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First user message"},
        {"role": "assistant", "content": "First assistant response"},
        {"role": "user", "content": "Second user message"},
        {"role": "assistant", "content": "Second assistant response"},
        {"role": "user", "content": "Third user message"},
        {"role": "assistant", "content": "Third assistant response"},
    ]
    context = LLMContext(messages)
    user = NvidiaUserContextAggregator(context=context, chat_history_limit=2, preserve_prompt_messages=0)
    truncated_context = await user.get_truncated_context()
    # With preserve_prompt_messages=0, no initial messages are preserved via that mechanism
    # However, system messages are always preserved regardless of limit (per docstring)
    # Only the last 2 turns should be kept from the remaining messages
    assert truncated_context.get_messages() == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Second user message"},
        {"role": "assistant", "content": "Second assistant response"},
        {"role": "user", "content": "Third user message"},
        {"role": "assistant", "content": "Third assistant response"},
    ]

    # Test case 2: preserve_prompt_messages=1 (default - preserve system message)
    context2 = LLMContext(messages)
    user2 = NvidiaUserContextAggregator(context=context2, chat_history_limit=2, preserve_prompt_messages=1)
    truncated_context2 = await user2.get_truncated_context()
    # With preserve_prompt_messages=1, first message (system) should be preserved
    assert truncated_context2.get_messages() == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Second user message"},
        {"role": "assistant", "content": "Second assistant response"},
        {"role": "user", "content": "Third user message"},
        {"role": "assistant", "content": "Third assistant response"},
    ]

    # Test case 3: preserve_prompt_messages=2 (Nemotron - preserve system + first user)
    context3 = LLMContext(messages)
    user3 = NvidiaUserContextAggregator(context=context3, chat_history_limit=2, preserve_prompt_messages=2)
    truncated_context3 = await user3.get_truncated_context()
    # With preserve_prompt_messages=2, first two messages should be preserved
    assert truncated_context3.get_messages() == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First user message"},
        {"role": "user", "content": "Second user message"},
        {"role": "assistant", "content": "Second assistant response"},
        {"role": "user", "content": "Third user message"},
        {"role": "assistant", "content": "Third assistant response"},
    ]

    # Test case 4: preserve_prompt_messages=2 with more messages
    messages4 = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First user message"},
        {"role": "assistant", "content": "First assistant response"},
        {"role": "user", "content": "Second user message"},
        {"role": "assistant", "content": "Second assistant response"},
        {"role": "user", "content": "Third user message"},
        {"role": "assistant", "content": "Third assistant response"},
        {"role": "user", "content": "Fourth user message"},
        {"role": "assistant", "content": "Fourth assistant response"},
    ]
    context4 = LLMContext(messages4)
    user4 = NvidiaUserContextAggregator(context=context4, chat_history_limit=2, preserve_prompt_messages=2)
    truncated_context4 = await user4.get_truncated_context()
    # First 2 messages preserved, then last 2 turns from remaining messages
    assert truncated_context4.get_messages() == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First user message"},
        {"role": "user", "content": "Third user message"},
        {"role": "assistant", "content": "Third assistant response"},
        {"role": "user", "content": "Fourth user message"},
        {"role": "assistant", "content": "Fourth assistant response"},
    ]

    # Test case 5: Empty context (edge case)
    context5 = LLMContext([])
    user5 = NvidiaUserContextAggregator(context=context5, chat_history_limit=2, preserve_prompt_messages=2)
    truncated_context5 = await user5.get_truncated_context()
    # Empty context should return empty context
    assert truncated_context5.get_messages() == []

    # Test case 6: preserve_prompt_messages greater than total messages
    messages6 = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First user message"},
    ]
    context6 = LLMContext(messages6)
    user6 = NvidiaUserContextAggregator(context=context6, chat_history_limit=2, preserve_prompt_messages=5)
    truncated_context6 = await user6.get_truncated_context()
    # All messages should be preserved if preserve_prompt_messages > total messages
    assert truncated_context6.get_messages() == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First user message"},
    ]
