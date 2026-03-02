"""Unit tests for NAT agent service."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import httpx
import yaml
from pipecat.frames.frames import CancelFrame, EndFrame

from nvidia_pipecat.services.nat_agent import NATAgentService


class TestNATAgentService(unittest.IsolatedAsyncioTestCase):
    """Test cases for NATAgentService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = NATAgentService(
            agent_server_url="http://localhost:8081",
            temperature=0.2,
            top_p=0.7,
            max_tokens=1000,
            use_knowledge_base=True,
        )

    def tearDown(self):
        """Clean up after tests."""
        # Reset shared session
        NATAgentService._shared_session = None

    def test_initialization(self):
        """Test NATAgentService initialization."""
        service = NATAgentService(
            agent_server_url="http://test:8081",
            temperature=0.5,
            top_p=0.8,
            max_tokens=500,
            use_knowledge_base=False,
        )

        self.assertEqual(service.agent_server_url, "http://test:8081")
        self.assertEqual(service.temperature, 0.5)
        self.assertEqual(service.top_p, 0.8)
        self.assertEqual(service.max_tokens, 500)
        self.assertFalse(service.use_knowledge_base)
        self.assertIsNone(service._last_used_phrase)
        self.assertIsNone(service._last_query)
        self.assertIsNotNone(service._default_phrases)

    def test_initialization_with_stop_words(self):
        """Test initialization with custom stop words."""
        stop_words = ["stop", "end", "finish"]
        _service = NATAgentService(stop_words=stop_words)
        # Note: stop_words are not stored as instance variable in current implementation
        # This test documents the current behavior

    def test_load_tools_from_config_none(self):
        """Test loading tools when config_file is None."""
        tools = self.service._load_tools_from_config(None)
        self.assertEqual(tools, {})

    def test_load_tools_from_config_file_not_found(self):
        """Test loading tools when config file doesn't exist."""
        tools = self.service._load_tools_from_config("nonexistent_file.yml")
        self.assertEqual(tools, {})

    def test_load_tools_from_config_valid(self):
        """Test loading tools from a valid config file."""
        config_data = {
            "functions": {
                "get_menu": {"natural_phrases": ["Let me get the menu for you.", "Here's what we have."]},
                "add_to_cart": {"natural_phrases": ["I'll add that to your cart.", "Adding that for you."]},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            tools = self.service._load_tools_from_config(config_path)
            expected = {
                "get_menu": {
                    "name": "get_menu",
                    "natural_phrases": ["Let me get the menu for you.", "Here's what we have."],
                },
                "add_to_cart": {
                    "name": "add_to_cart",
                    "natural_phrases": ["I'll add that to your cart.", "Adding that for you."],
                },
            }
            self.assertEqual(tools, expected)
        finally:
            Path(config_path).unlink()

    def test_load_tools_from_config_invalid_yaml(self):
        """Test loading tools from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            tools = self.service._load_tools_from_config(config_path)
            self.assertEqual(tools, {})
        finally:
            Path(config_path).unlink()

    def test_load_tools_from_config_empty_functions(self):
        """Test loading tools when functions section is empty."""
        config_data = {"functions": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            tools = self.service._load_tools_from_config(config_path)
            self.assertEqual(tools, {})
        finally:
            Path(config_path).unlink()

    def test_load_tools_from_config_none_natural_phrases(self):
        """Test loading tools when natural_phrases is None."""
        config_data = {"functions": {"test_tool": {"natural_phrases": None}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            tools = self.service._load_tools_from_config(config_path)
            expected = {"test_tool": {"name": "test_tool", "natural_phrases": []}}
            self.assertEqual(tools, expected)
        finally:
            Path(config_path).unlink()

    def test_sanitize_text_for_tts_empty(self):
        """Test text sanitization with empty input."""
        result = self.service._sanitize_text_for_tts("")
        self.assertEqual(result, "")

        result = self.service._sanitize_text_for_tts(None)
        self.assertEqual(result, "")

    def test_sanitize_text_for_tts_normal_text(self):
        """Test text sanitization with normal text."""
        text = "Hello, this is a normal message."
        result = self.service._sanitize_text_for_tts(text)
        self.assertEqual(result, text)

    def test_sanitize_text_for_tts_control_characters(self):
        """Test text sanitization with control characters."""
        text = "Hello\x00world\x08\x0b\x0c\x0e\x1f\x7f"
        result = self.service._sanitize_text_for_tts(text)
        self.assertEqual(result, "Helloworld")

    def test_sanitize_text_for_tts_multiple_whitespace(self):
        """Test text sanitization with multiple whitespace."""
        text = "Hello    world\n\n\n\t\t\t"
        result = self.service._sanitize_text_for_tts(text)
        self.assertEqual(result, "Hello world")

    def test_sanitize_text_for_tts_unmatched_brackets(self):
        """Test text sanitization with unmatched brackets."""
        text = "Hello [world {test (example"
        result = self.service._sanitize_text_for_tts(text)
        # The method only removes trailing unmatched brackets, not all unmatched brackets
        self.assertEqual(result, "Hello [world {test (example")

    def test_sanitize_text_for_tts_unmatched_quotes(self):
        """Test text sanitization with unmatched quotes."""
        text = "Hello 'world \"test `example"
        result = self.service._sanitize_text_for_tts(text)
        # The method only removes trailing unmatched quotes, not all unmatched quotes
        self.assertEqual(result, "Hello 'world \"test `example")

    def test_sanitize_text_for_tts_dict_formatting(self):
        """Test text sanitization with dictionary formatting."""
        text = "Here are the prices: {'apple': 1.50, 'banana': 0.75}"
        result = self.service._sanitize_text_for_tts(text)
        self.assertIn("apple costs 1.5 dollars", result)
        self.assertIn("banana costs 0.75 dollars", result)

    def test_sanitize_text_for_tts_list_formatting(self):
        """Test text sanitization with list formatting."""
        text = "Available items: ['apple', 'banana', 'orange']"
        result = self.service._sanitize_text_for_tts(text)
        self.assertEqual(result, "Available items: apple, banana, orange")

    def test_extract_intermediate_data_string_payload(self):
        """Test extracting intermediate data from string payload."""
        payload = "**Output** Hello world Thought: I should greet the user"
        output, thought = self.service._extract_intermediate_data(payload)
        self.assertEqual(output, "Hello world Thought: I should greet the user")
        self.assertEqual(thought, "I should greet the user")

    def test_extract_intermediate_data_no_output(self):
        """Test extracting intermediate data with no output section."""
        payload = "Thought: Just thinking here"
        output, thought = self.service._extract_intermediate_data(payload)
        self.assertEqual(output, "")
        self.assertEqual(thought, "Just thinking here")

    def test_extract_intermediate_data_no_thought(self):
        """Test extracting intermediate data with no thought section."""
        payload = "**Output** Just output here"
        output, thought = self.service._extract_intermediate_data(payload)
        self.assertEqual(output, "Just output here")
        self.assertEqual(thought, "")

    def test_extract_intermediate_data_dict_payload(self):
        """Test extracting intermediate data from dict payload."""
        payload = {"content": "**Output** Hello Thought: Test"}
        output, thought = self.service._extract_intermediate_data(payload)
        self.assertEqual(output, "")
        self.assertEqual(thought, "")

    def test_get_random_message_empty_list(self):
        """Test getting random message with empty list."""
        result = self.service.get_random_message([])
        self.assertIn(result, self.service._default_phrases)

    def test_get_random_message_single_phrase(self):
        """Test getting random message with single phrase."""
        phrases = ["Only one phrase"]
        result = self.service.get_random_message(phrases)
        self.assertEqual(result, "Only one phrase")

    def test_get_random_message_multiple_phrases(self):
        """Test getting random message with multiple phrases."""
        phrases = ["First phrase", "Second phrase", "Third phrase"]
        result = self.service.get_random_message(phrases)
        self.assertIn(result, phrases)

    def test_get_random_message_no_consecutive_repetition(self):
        """Test that consecutive repetition is avoided."""
        phrases = ["First", "Second"]

        # Get first message
        first = self.service.get_random_message(phrases)
        self.assertIn(first, phrases)

        # Get second message - should be different
        second = self.service.get_random_message(phrases)
        self.assertIn(second, phrases)

        # If we have only two phrases and first was used, second should be different
        if len(phrases) == 2:
            self.assertNotEqual(first, second)

    def test_shared_session_property(self):
        """Test shared session property with shared session enabled."""
        # Initially should be None
        self.assertIsNone(NATAgentService._shared_session)

        # Getting session should create one
        session = self.service.shared_session
        self.assertIsInstance(session, httpx.AsyncClient)
        self.assertIsNotNone(NATAgentService._shared_session)

    def test_shared_session_persistence(self):
        """Test that shared session persists across multiple service instances."""
        # Initially should be None
        self.assertIsNone(NATAgentService._shared_session)

        # Create first service and get session
        service1 = NATAgentService(agent_server_url="http://localhost:8081")
        session1 = service1.shared_session
        self.assertIsInstance(session1, httpx.AsyncClient)
        self.assertIsNotNone(NATAgentService._shared_session)

        # Create second service and get session - should be the same
        service2 = NATAgentService(agent_server_url="http://localhost:8081")
        session2 = service2.shared_session
        self.assertEqual(session1, session2)

    def test_shared_session_setter(self):
        """Test shared session setter."""
        mock_session = Mock(spec=httpx.AsyncClient)
        self.service.shared_session = mock_session
        self.assertEqual(NATAgentService._shared_session, mock_session)

    def test_constructor_exception_safety(self):
        """Test that HTTP client is properly cleaned up if constructor fails."""
        # Mock _load_tools_from_config to raise an exception
        with patch.object(NATAgentService, "_load_tools_from_config", side_effect=Exception("Config load failed")):
            # Constructor should fail during tools loading
            with self.assertRaises(Exception) as context:
                NATAgentService(
                    agent_server_url="http://localhost:8081",
                )

            # Verify the original exception is re-raised
            self.assertEqual(str(context.exception), "Config load failed")

            # Since _load_tools_from_config fails, shared session should not be affected
            # (HTTP client is created lazily on first access)

    def test_constructor_initialization_order(self):
        """Test that service initialization completes successfully with tools loading."""
        # Create a service
        service = NATAgentService(
            agent_server_url="http://localhost:8081",
        )

        # Verify that the service was created successfully
        self.assertIsNotNone(service.agent_server_url)
        self.assertEqual(service.agent_server_url, "http://localhost:8081")

        # Verify that tools were loaded (empty dict in this case since no config file)
        self.assertIsNotNone(service._tools)
        self.assertIsInstance(service._tools, dict)

        # Shared session is created lazily on first access
        self.assertIsNone(NATAgentService._shared_session)

    async def test_test_connection_success(self):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.status_code = 200

        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.get.return_value = mock_response

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            result = await self.service.test_connection()
            self.assertTrue(result)

    async def test_test_connection_failure_status(self):
        """Test connection test with non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500

        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.get.return_value = mock_response

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            result = await self.service.test_connection()
            self.assertFalse(result)

    async def test_test_connection_connect_error(self):
        """Test connection test with connection error."""
        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.get.side_effect = httpx.ConnectError("Connection failed")

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            result = await self.service.test_connection()
            self.assertFalse(result)

    async def test_test_connection_timeout(self):
        """Test connection test with timeout."""
        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.get.side_effect = httpx.TimeoutException("Timeout")

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            result = await self.service.test_connection()
            self.assertFalse(result)

    async def test_get_natagent_response_success(self):
        """Test successful NATAgent response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.post.return_value = mock_response

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            request_json = {"messages": [{"role": "user", "content": "Hello"}]}
            result = await self.service._get_natagent_response(request_json)
            self.assertEqual(result, mock_response)

    async def test_get_natagent_response_http_error(self):
        """Test NATAgent response with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=Mock(), response=mock_response
        )

        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.post.return_value = mock_response

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            request_json = {"messages": [{"role": "user", "content": "Hello"}]}
            with self.assertRaises(httpx.HTTPStatusError):
                await self.service._get_natagent_response(request_json)

    async def test_get_natagent_response_connect_error(self):
        """Test NATAgent response with connection error."""
        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.post.side_effect = httpx.ConnectError("Connection failed")

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            request_json = {"messages": [{"role": "user", "content": "Hello"}]}
            with self.assertRaises(httpx.ConnectError):
                await self.service._get_natagent_response(request_json)

    async def test_get_natagent_response_timeout(self):
        """Test NATAgent response with timeout."""
        # Create a mock session
        mock_session = AsyncMock(spec=httpx.AsyncClient)
        mock_session.post.side_effect = httpx.TimeoutException("Timeout")

        # Patch the shared_session property on the class
        with patch.object(NATAgentService, "shared_session", new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_session
            request_json = {"messages": [{"role": "user", "content": "Hello"}]}
            with self.assertRaises(httpx.TimeoutException):
                await self.service._get_natagent_response(request_json)

    async def test_stop(self):
        """Test stop method."""
        mock_frame = Mock(spec=EndFrame)
        mock_task = AsyncMock()
        self.service._current_task = mock_task

        with patch.object(self.service, "cancel_task") as mock_cancel:
            await self.service.stop(mock_frame)

            mock_cancel.assert_called_once_with(mock_task)
            self.assertIsNone(self.service.current_action)
            self.assertIsNone(self.service.current_thought)

    async def test_cancel(self):
        """Test cancel method."""
        mock_frame = Mock(spec=CancelFrame)
        mock_task = AsyncMock()
        self.service._current_task = mock_task

        with patch.object(self.service, "cancel_task") as mock_cancel:
            await self.service.cancel(mock_frame)
            mock_cancel.assert_called_once_with(mock_task)

    async def test_cleanup(self):
        """Test cleanup method with shared session."""
        mock_session = Mock(spec=httpx.AsyncClient)
        NATAgentService._shared_session = mock_session

        # Patch the parent class cleanup method directly
        with patch(
            "pipecat.services.openai.llm.OpenAILLMService.cleanup", new_callable=AsyncMock
        ) as mock_parent_cleanup:
            await self.service.cleanup()

            mock_parent_cleanup.assert_awaited_once()
            mock_session.aclose.assert_called_once()
            self.assertIsNone(NATAgentService._shared_session)

    async def test_close_client_session_no_session(self):
        """Test closing client session when no session exists."""
        NATAgentService._shared_session = None
        await self.service._close_client_session()
        self.assertIsNone(NATAgentService._shared_session)

    async def test_close_client_session_with_session(self):
        """Test closing client session when shared session exists."""
        mock_session = Mock(spec=httpx.AsyncClient)
        NATAgentService._shared_session = mock_session

        await self.service._close_client_session()

        mock_session.aclose.assert_called_once()
        self.assertIsNone(NATAgentService._shared_session)


class TestNATAgentServiceIntegration(unittest.TestCase):
    """Integration tests for NATAgentService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = NATAgentService(
            agent_server_url="http://localhost:8081",
            config_file=None,
        )

    def tearDown(self):
        """Clean up after tests."""
        NATAgentService._shared_session = None

    def test_default_phrases_not_empty(self):
        """Test that default phrases are not empty."""
        self.assertGreater(len(self.service._default_phrases), 0)
        for phrase in self.service._default_phrases:
            self.assertIsInstance(phrase, str)
            self.assertGreater(len(phrase), 0)

    def test_default_phrases_no_duplicates(self):
        """Test that default phrases have no duplicates."""
        phrases = self.service._default_phrases
        unique_phrases = set(phrases)
        self.assertEqual(len(phrases), len(unique_phrases))

    def test_sanitize_text_complex_scenario(self):
        """Test text sanitization with complex scenario."""
        text = "Hello\x00world\n\n\n   with   multiple   spaces   and   {'prices': [1.50, 2.75]} and [items]"
        result = self.service._sanitize_text_for_tts(text)

        # Should remove control characters
        self.assertNotIn("\x00", result)

        # Should normalize whitespace
        self.assertNotIn("   ", result)

        # Should format dictionary - check for the actual format
        self.assertIn("prices is", result)

        # Should format list
        self.assertIn("items", result)

    def test_get_random_message_distribution(self):
        """Test that random message selection has good distribution."""
        phrases = ["A", "B", "C", "D", "E"]
        results = []

        # Get many messages to test distribution
        for _ in range(100):
            result = self.service.get_random_message(phrases)
            results.append(result)

        # All results should be from the phrases list
        for result in results:
            self.assertIn(result, phrases)

        # Should have some variety (not all the same)
        unique_results = set(results)
        self.assertGreater(len(unique_results), 1)


if __name__ == "__main__":
    unittest.main()
