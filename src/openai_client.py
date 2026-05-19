"""Small OpenAI review client wrapper for backend LLM supplier review.

No API key is hardcoded here. Live mode uses OPENAI_API_KEY from config/env.
The wrapper produces strict JSON requests and stores request/response JSONL so
the backend can audit every LLM-mediated decision.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


LLM_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_group_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["approve", "reject", "split", "merge_with_existing", "promote_score", "uncertain"]},
        "final_score": {"type": ["integer", "null"], "enum": [85, 98, None]},
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "row_ids": {"type": "array", "items": {"type": "integer"}},
                    "final_score": {"type": ["integer", "null"], "enum": [85, 98, None]},
                    "reasoning": {"type": "string"},
                },
                "required": ["row_ids", "final_score", "reasoning"],
            },
        },
        "rejected_row_ids": {"type": "array", "items": {"type": "integer"}},
        "target_cluster_number": {"type": ["integer", "null"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "candidate_group_id",
        "decision",
        "final_score",
        "clusters",
        "rejected_row_ids",
        "target_cluster_number",
        "reasoning",
        "confidence",
    ],
}


SYSTEM_PROMPT = """You review supplier/vendor candidate groups for procurement cleanup.
Return JSON only. Do not invent facts. Apply hard rules: reject city-only,
generic-only, first-name/surname-only, hotel/franchise brand-only, protected
compound conflicts, and tax/address/domain conflicts. Use approve only when the
rows plausibly represent the same supplier brand/group identity. Use split when
only subgroups belong together. Use uncertain when evidence is insufficient."""


def build_openai_request(group: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Build a structured-output Responses API request body."""
    user_payload = {
        "candidate_group_id": group.get("candidate_group_id"),
        "priority": group.get("priority"),
        "row_ids": group.get("row_ids", []),
        "records": group.get("records", []),
        "deterministic_pass_types": group.get("deterministic_pass_types", []),
        "raw_scores": group.get("raw_scores", []),
        "reason_for_llm_review": group.get("reason_for_llm_review", ""),
        "hard_rules": [
            "Do not approve city/location-only matches.",
            "Do not approve generic-token-only matches.",
            "Do not approve first-name-only or surname-only individual matches.",
            "Do not approve hotel/restaurant/franchise brand-only matches.",
            "Do not merge protected compounds such as Eastman Kodak into Eastman Chemical without explicit evidence.",
            "Do not merge Air Liquide with Air Products.",
            "Final score may only be 85, 98, or null.",
        ],
    }
    return {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "supplier_llm_decision",
                "schema": LLM_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    }


class OpenAIReviewClient:
    def __init__(self, api_key: str, model: str = "gpt-5.5", base_url: str = "https://api.openai.com/v1", timeout: int = 60, retries: int = 2):
        self.api_key = api_key or ""
        self.model = model or "gpt-5.5"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = int(timeout or 60)
        self.retries = int(retries or 0)

    def review_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        body = build_openai_request(group, self.model)
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return parse_openai_response(payload)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenAI review failed: {last_error}")


def parse_openai_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract strict JSON from common Responses API shapes."""
    if "output_text" in payload:
        return json.loads(payload["output_text"])
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if text:
                return json.loads(text)
    raise RuntimeError("OpenAI response did not contain JSON text")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

