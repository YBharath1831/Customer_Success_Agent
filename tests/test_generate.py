"""Tests for structured and cited response generation."""

import json
from dataclasses import dataclass

import pytest
import requests

from src.generate import (
    GeneratedDraft,
    GenerationError,
    OpenRouterClient,
    OpenRouterConfig,
    ResponseGenerator,
    SYSTEM_PROMPT_PATH,
    load_system_prompt,
)
from src.ingest import normalize_ticket


@dataclass(frozen=True)
class PassageStub:
    passage_id: str
    text: str


class FakeGenerationClient:
    """Deterministic model client that makes no network requests."""

    def __init__(
        self,
        response: str,
        model_name: str = "fake-test-model",
    ) -> None:
        self.response = response
        self._model_name = model_name
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self.response


class FakeHTTPResponse:
    """Small replacement for requests.Response."""

    def __init__(
        self,
        *,
        status_code: int,
        payload=None,
        invalid_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("invalid JSON")

        return self.payload


class FakeSession:
    """Fake requests session with queued responses."""

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(
        self,
        url,
        *,
        headers,
        json,
        timeout,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )

        if not self.responses:
            raise AssertionError(
                "No fake HTTP responses remain."
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def make_ticket():
    return normalize_ticket(
        {
            "ticket_id": "GEN-001",
            "channel": "email",
            "subject": "Password reset",
            "body": "How do I reset my password?",
        }
    )


def make_passage(
    passage_id: str = "DOC001#p001",
) -> PassageStub:
    return PassageStub(
        passage_id=passage_id,
        text=(
            "Users can reset their password from the account "
            "settings page."
        ),
    )


def valid_model_response() -> str:
    return json.dumps(
        {
            "answer": (
                "You can reset your password from the account "
                "settings page [DOC001#p001]."
            ),
            "claims": [
                {
                    "text": (
                        "Users can reset their password from the "
                        "account settings page."
                    ),
                    "passage_ids": ["DOC001#p001"],
                }
            ],
        }
    )


def test_generator_returns_structured_draft() -> None:
    client = FakeGenerationClient(
        valid_model_response()
    )
    generator = ResponseGenerator(client)

    draft = generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=[make_passage()],
    )

    assert isinstance(draft, GeneratedDraft)
    assert draft.model == "fake-test-model"
    assert draft.answer_text.endswith(
        "[DOC001#p001]."
    )
    assert len(draft.claims) == 1
    assert draft.claims[0].passage_ids == (
        "DOC001#p001",
    )
    assert draft.cited_passage_ids == (
        "DOC001#p001",
    )


def test_prompt_contains_ticket_and_documentation() -> None:
    client = FakeGenerationClient(
        valid_model_response()
    )
    generator = ResponseGenerator(client)

    generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=[make_passage()],
    )

    assert len(client.system_prompts) == 1
    assert len(client.user_prompts) == 1

    system_prompt = client.system_prompts[0]
    user_prompt = client.user_prompts[0]

    assert "Use only facts" in system_prompt
    assert "untrusted data" in system_prompt

    assert "GEN-001" in user_prompt
    assert "Password reset" in user_prompt
    assert "How do I reset my password?" in user_prompt
    assert "account_access" in user_prompt
    assert "DOC001#p001" in user_prompt
    assert "Users can reset their password" in user_prompt


def test_versioned_system_prompt_is_available() -> None:
    prompt = load_system_prompt()

    assert SYSTEM_PROMPT_PATH.name == (
        "response_generator_v1.txt"
    )
    assert "Use only facts" in prompt
    assert "untrusted data" in prompt


def test_missing_system_prompt_is_rejected(
    tmp_path,
) -> None:
    with pytest.raises(
        GenerationError,
        match="could not be loaded",
    ):
        load_system_prompt(
            tmp_path / "missing-prompt.txt"
        )


def test_empty_system_prompt_is_rejected(
    tmp_path,
) -> None:
    prompt_path = tmp_path / "empty-prompt.txt"
    prompt_path.write_text("   ", encoding="utf-8")

    with pytest.raises(
        GenerationError,
        match="is empty",
    ):
        load_system_prompt(prompt_path)


def test_json_inside_markdown_fence_is_accepted() -> None:
    fenced_response = (
        "```json\n"
        f"{valid_model_response()}\n"
        "```"
    )

    generator = ResponseGenerator(
        FakeGenerationClient(fenced_response)
    )

    draft = generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=[make_passage()],
    )

    assert draft.answer_text
    assert draft.claims


def test_json_with_surrounding_text_is_recovered() -> None:
    response = (
        "Here is the requested JSON:\n"
        f"{valid_model_response()}\n"
        "End of response."
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    draft = generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=[make_passage()],
    )

    assert draft.model == "fake-test-model"


def test_invalid_json_is_rejected() -> None:
    generator = ResponseGenerator(
        FakeGenerationClient(
            "This response is not JSON."
        )
    )

    with pytest.raises(
        GenerationError,
        match="valid JSON",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_missing_answer_is_rejected() -> None:
    response = json.dumps(
        {
            "claims": [
                {
                    "text": "Password reset is available.",
                    "passage_ids": ["DOC001#p001"],
                }
            ]
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="missing a string answer",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_empty_answer_is_rejected() -> None:
    response = json.dumps(
        {
            "answer": "   ",
            "claims": [
                {
                    "text": "Password reset is available.",
                    "passage_ids": ["DOC001#p001"],
                }
            ],
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="answer is empty",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


@pytest.mark.parametrize(
    "claims_value",
    [
        None,
        [],
        "not-a-list",
    ],
)
def test_missing_or_invalid_claims_are_rejected(
    claims_value,
) -> None:
    response = json.dumps(
        {
            "answer": (
                "Reset your password [DOC001#p001]."
            ),
            "claims": claims_value,
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="at least one claim",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_claim_without_passage_ids_is_rejected() -> None:
    response = json.dumps(
        {
            "answer": (
                "Reset your password [DOC001#p001]."
            ),
            "claims": [
                {
                    "text": "Reset your password.",
                    "passage_ids": [],
                }
            ],
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="has no citations",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_unknown_answer_citation_is_rejected() -> None:
    response = json.dumps(
        {
            "answer": (
                "Reset your password [DOC999#p999]."
            ),
            "claims": [
                {
                    "text": "Reset your password.",
                    "passage_ids": ["DOC999#p999"],
                }
            ],
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="unknown citations",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_unknown_claim_citation_is_rejected() -> None:
    response = json.dumps(
        {
            "answer": (
                "Reset your password [DOC001#p001]."
            ),
            "claims": [
                {
                    "text": "Reset your password.",
                    "passage_ids": ["DOC999#p999"],
                }
            ],
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="unknown passage",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_claim_citation_must_appear_in_answer() -> None:
    response = json.dumps(
        {
            "answer": (
                "You can reset your password from account settings."
            ),
            "claims": [
                {
                    "text": (
                        "You can reset your password from "
                        "account settings."
                    ),
                    "passage_ids": ["DOC001#p001"],
                }
            ],
        }
    )

    generator = ResponseGenerator(
        FakeGenerationClient(response)
    )

    with pytest.raises(
        GenerationError,
        match="does not appear",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[make_passage()],
        )


def test_generation_without_passages_is_rejected() -> None:
    generator = ResponseGenerator(
        FakeGenerationClient(
            valid_model_response()
        )
    )

    with pytest.raises(
        GenerationError,
        match="without passages",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=[],
        )


def test_duplicate_passage_ids_are_rejected() -> None:
    generator = ResponseGenerator(
        FakeGenerationClient(
            valid_model_response()
        )
    )

    passages = [
        make_passage("DOC001#p001"),
        PassageStub(
            passage_id="DOC001#p001",
            text="A different passage with the same ID.",
        ),
    ]

    with pytest.raises(
        GenerationError,
        match="Duplicate retrieved passage ID",
    ):
        generator.generate(
            ticket=make_ticket(),
            intent="account_access",
            urgency="normal",
            passages=passages,
        )


def test_generator_limits_number_of_passages() -> None:
    client = FakeGenerationClient(
        valid_model_response()
    )

    generator = ResponseGenerator(
        client,
        maximum_passages=1,
    )

    passages = [
        make_passage("DOC001#p001"),
        PassageStub(
            passage_id="DOC002#p001",
            text="This passage should not enter the prompt.",
        ),
    ]

    generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=passages,
    )

    user_prompt = client.user_prompts[0]

    assert "DOC001#p001" in user_prompt
    assert "DOC002#p001" not in user_prompt
    assert "should not enter the prompt" not in user_prompt


def test_passage_text_is_truncated() -> None:
    client = FakeGenerationClient(
        valid_model_response()
    )

    generator = ResponseGenerator(
        client,
        maximum_passage_characters=20,
    )

    generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=[make_passage()],
    )

    user_prompt = client.user_prompts[0]

    assert "Users can reset thei" in user_prompt
    assert "account settings page" not in user_prompt


def test_generated_draft_serializes() -> None:
    generator = ResponseGenerator(
        FakeGenerationClient(
            valid_model_response()
        )
    )

    draft = generator.generate(
        ticket=make_ticket(),
        intent="account_access",
        urgency="normal",
        passages=[make_passage()],
    )

    serialized = draft.as_dict()

    assert serialized["model"] == "fake-test-model"
    assert serialized["cited_passage_ids"] == [
        "DOC001#p001"
    ]
    assert serialized["claims"][0]["passage_ids"] == [
        "DOC001#p001"
    ]


@pytest.mark.parametrize(
    ("argument_name", "argument_value"),
    [
        ("maximum_passages", 0),
        ("maximum_passage_characters", 0),
    ],
)
def test_invalid_generator_configuration_is_rejected(
    argument_name: str,
    argument_value: int,
) -> None:
    arguments = {
        "maximum_passages": 5,
        "maximum_passage_characters": 4000,
    }
    arguments[argument_name] = argument_value

    with pytest.raises(ValueError):
        ResponseGenerator(
            FakeGenerationClient(
                valid_model_response()
            ),
            **arguments,
        )


def test_openrouter_config_validates_values() -> None:
    with pytest.raises(
        ValueError,
        match="API key",
    ):
        OpenRouterConfig(
            api_key="",
            model="test-model",
        )

    with pytest.raises(
        ValueError,
        match="model",
    ):
        OpenRouterConfig(
            api_key="test-key",
            model="",
        )

    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        OpenRouterConfig(
            api_key="test-key",
            model="test-model",
            timeout_seconds=0,
        )

    with pytest.raises(
        ValueError,
        match="max_retries",
    ):
        OpenRouterConfig(
            api_key="test-key",
            model="test-model",
            max_retries=-1,
        )


def test_openrouter_client_extracts_content() -> None:
    session = FakeSession(
        [
            FakeHTTPResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": valid_model_response()
                            }
                        }
                    ]
                },
            )
        ]
    )

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-secret-key",
            model="test-model",
            timeout_seconds=12,
            max_retries=0,
        ),
        session=session,
    )

    content = client.complete(
        system_prompt="system",
        user_prompt="user",
    )

    assert content == valid_model_response()
    assert len(session.calls) == 1
    assert session.calls[0]["timeout"] == 12
    assert session.calls[0]["json"]["model"] == "test-model"
    assert (
        session.calls[0]["headers"]["Authorization"]
        == "Bearer test-secret-key"
    )


def test_openrouter_client_retries_server_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.generate.time.sleep",
        lambda _: None,
    )

    session = FakeSession(
        [
            FakeHTTPResponse(
                status_code=500,
                payload={},
            ),
            FakeHTTPResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": valid_model_response()
                            }
                        }
                    ]
                },
            ),
        ]
    )

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-key",
            model="test-model",
            max_retries=1,
        ),
        session=session,
    )

    content = client.complete(
        system_prompt="system",
        user_prompt="user",
    )

    assert content == valid_model_response()
    assert len(session.calls) == 2


def test_openrouter_client_fails_after_timeout_retries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.generate.time.sleep",
        lambda _: None,
    )

    session = FakeSession(
        [
            requests.Timeout("first timeout"),
            requests.Timeout("second timeout"),
        ]
    )

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-key",
            model="test-model",
            max_retries=1,
        ),
        session=session,
    )

    with pytest.raises(
        GenerationError,
        match="after 2 attempt",
    ):
        client.complete(
            system_prompt="system",
            user_prompt="user",
        )

    assert len(session.calls) == 2


def test_openrouter_client_does_not_retry_client_error() -> None:
    session = FakeSession(
        [
            FakeHTTPResponse(
                status_code=401,
                payload={
                    "error": "Unauthorized",
                },
            )
        ]
    )

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="do-not-expose-this-key",
            model="test-model",
            max_retries=3,
        ),
        session=session,
    )

    with pytest.raises(
        GenerationError,
        match="status 401",
    ) as error:
        client.complete(
            system_prompt="system",
            user_prompt="user",
        )

    assert len(session.calls) == 1
    assert "do-not-expose-this-key" not in str(error.value)


def test_openrouter_invalid_json_is_rejected() -> None:
    session = FakeSession(
        [
            FakeHTTPResponse(
                status_code=200,
                invalid_json=True,
            )
        ]
    )

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-key",
            model="test-model",
            max_retries=0,
        ),
        session=session,
    )

    with pytest.raises(
        GenerationError,
        match="invalid JSON",
    ):
        client.complete(
            system_prompt="system",
            user_prompt="user",
        )
