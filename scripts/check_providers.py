"""Offline provider routing and HTTP contract checks. No live API requests."""
import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app import llm
from app.compatible_llm import CompatibleInterpreter

PROPOSAL = {"outcome": "propose", "message": "List tasks", "actions": [{"operation": "list_tasks"}]}


def completion(content=None, finish="stop"):
    return httpx.Response(200, json={"choices": [{
        "finish_reason": finish,
        "message": {"content": json.dumps(PROPOSAL) if content is None else content},
    }]})


class Providers(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        llm.get_interpreter.cache_clear()
        self.sleep = patch("app.compatible_llm.time.sleep")
        self.sleep.start()

    def tearDown(self):
        llm.get_interpreter.cache_clear()
        self.sleep.stop()
        self.env.stop()

    def test_legacy_groq(self):
        os.environ.update(GROQ_API_KEY="legacy", GROQ_MODEL="legacy-model")
        with patch.object(llm, "ChatGroq") as groq:
            llm.get_interpreter()
            self.assertEqual(groq.call_args.kwargs["api_key"], "legacy")
            self.assertEqual(groq.call_args.kwargs["model"], "legacy-model")
            groq.return_value.with_structured_output.assert_called_once_with(llm.AgentProposal, method="json_mode")

    def test_generic_groq_settings_take_precedence(self):
        os.environ.update(GROQ_API_KEY="old", LLM_API_KEY="new", LLM_MODEL="new-model")
        with patch.object(llm, "ChatGroq") as groq:
            llm.get_interpreter()
            self.assertEqual(groq.call_args.kwargs["api_key"], "new")
            self.assertEqual(groq.call_args.kwargs["model"], "new-model")

    def test_openai_route_does_not_use_groq_key_or_custom_url(self):
        os.environ.update(LLM_PROVIDER="openai", OPENAI_API_KEY="openai-key", LLM_MODEL="model", GROQ_API_KEY="old", LLM_BASE_URL="https://other.example/v1")
        result = llm.get_interpreter()
        self.assertEqual(result.endpoint, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(result.api_key, "openai-key")

    def test_missing_and_invalid_settings(self):
        cases = [{}, {"LLM_PROVIDER": "unknown"},
                 {"LLM_PROVIDER": "openai", "GROQ_API_KEY": "wrong", "LLM_MODEL": "model"},
                 {"LLM_PROVIDER": "openai_compatible", "LLM_API_KEY": "key", "LLM_MODEL": "model"}]
        for settings in cases:
            with self.subTest(settings=settings), patch.dict(os.environ, settings, clear=True):
                llm.get_interpreter.cache_clear()
                with self.assertRaises(RuntimeError):
                    llm.get_interpreter()

    def test_url_validation(self):
        for url in ["http://external.example/v1", "https://user:secret@example.com/v1", "https://example.com/v1?key=x", "ftp://example.com", ""]:
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                CompatibleInterpreter("key", "model", url)
        self.assertEqual(CompatibleInterpreter("key", "model", "http://localhost:1234/v1/").endpoint, "http://localhost:1234/v1/chat/completions")

    def test_compatible_wire_format_and_validation(self):
        os.environ.update(LLM_PROVIDER="openai_compatible", LLM_API_KEY="key", LLM_MODEL="model", LLM_BASE_URL="https://provider.example/v1/")
        with patch("app.compatible_llm.httpx.Client") as client:
            post = client.return_value.__enter__.return_value.post
            post.return_value = completion()
            result = llm.get_interpreter().invoke([("system", "JSON"), ("human", "list"), ("assistant", "reply")])
            self.assertEqual(result.actions[0].operation, "list_tasks")
            self.assertEqual(post.call_args.args[0], "https://provider.example/v1/chat/completions")
            data = post.call_args.kwargs["json"]
            self.assertEqual([m["role"] for m in data["messages"]], ["system", "user", "assistant"])
            self.assertEqual(data["response_format"], {"type": "json_object"})
            client.assert_called_once_with(timeout=30, follow_redirects=False)

    def test_invalid_proposal_is_rejected(self):
        with patch("app.compatible_llm.httpx.Client.post", return_value=completion('{"outcome":"propose","actions":[]}')):
            with self.assertRaises(ValidationError):
                CompatibleInterpreter("key", "model", "https://example.com/v1").invoke([])

    def test_incomplete_response_is_rejected(self):
        with patch("app.compatible_llm.httpx.Client.post", return_value=completion(finish="length")):
            with self.assertRaises(OutputParserException):
                CompatibleInterpreter("key", "model", "https://example.com/v1").invoke([])

    def test_auth_error_does_not_retry_or_echo_body(self):
        with patch("app.compatible_llm.httpx.Client.post", return_value=httpx.Response(401, text="sensitive body")) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
                CompatibleInterpreter("key", "model", "https://example.com/v1").invoke([])
            self.assertNotIn("sensitive", str(error.exception))
            self.assertEqual(post.call_count, 1)

    def test_transient_errors_retry_once(self):
        for failure in [httpx.Response(429), httpx.Response(503), httpx.ConnectError("offline")]:
            with self.subTest(failure=failure), patch("app.compatible_llm.httpx.Client.post", side_effect=[failure, completion()]) as post:
                result = CompatibleInterpreter("key", "model", "https://example.com/v1").invoke([])
                self.assertEqual(result.outcome, "propose")
                self.assertEqual(post.call_count, 2)

    def test_format_retry_uses_existing_interpretation_flow(self):
        os.environ.update(LLM_PROVIDER="openai_compatible", LLM_API_KEY="key", LLM_MODEL="model", LLM_BASE_URL="https://example.com/v1")
        with patch("app.compatible_llm.httpx.Client.post", side_effect=[completion("not json"), completion()]) as post:
            self.assertEqual(llm.interpret_message("list tasks").outcome, "propose")
            self.assertEqual(post.call_count, 2)
            self.assertIn("response failed validation", post.call_args.kwargs["json"]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
