"""Chat Completions adapter; all returned task proposals remain validated."""

import time
from urllib.parse import urlsplit

import httpx
from langchain_core.exceptions import OutputParserException

from app.agent_schemas import AgentProposal


class CompatibleInterpreter:
    def __init__(self, api_key: str, model: str, base_url: str):
        parsed = urlsplit(base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment
                or not (parsed.scheme == "https" or (local and parsed.scheme == "http"))):
            raise RuntimeError("LLM_BASE_URL must be an HTTPS API base URL (HTTP is allowed for localhost).")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model

    def invoke(self, messages):
        roles = {"human": "user", "assistant": "assistant", "system": "system"}
        payload = {
            "model": self.model,
            "messages": [{"role": roles[role], "content": content} for role, content in messages],
            "response_format": {"type": "json_object"},
        }
        # Do not forward authorization to redirects or silently switch providers.
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            for attempt in range(2):
                try:
                    response = client.post(
                        self.endpoint, json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                except httpx.TransportError:
                    if attempt == 0:
                        time.sleep(1)
                        continue
                    raise RuntimeError("LLM connection failed. Check the endpoint and network, then retry.") from None
                if attempt == 0 and (response.status_code == 429 or response.status_code >= 500):
                    time.sleep(1)
                    continue
                if not response.is_success:
                    raise RuntimeError(
                        f"LLM request failed (HTTP {response.status_code}). Check the provider, key, "
                        "model and base URL; the model must support Chat Completions JSON mode."
                    )
                break

        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") != "stop" or not isinstance(content, str):
                raise ValueError("Incomplete or refused response")
        except (ValueError, KeyError, IndexError, TypeError):
            raise OutputParserException("LLM returned an incomplete or invalid Chat Completions response.") from None
        return AgentProposal.model_validate_json(content)
