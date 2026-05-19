"""AI-risk review module for ambiguous candidate pairs.

This module is intentionally called only for risky candidates, not all rows.
It supports OpenAI-compatible chat APIs, Claude/Anthropic style APIs, and Gemini.
"""
import json
import os
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import httpx

@dataclass
class AIReviewResult:
    verdict: str  # SAME_ENTITY, RELATED, DIFFERENT, UNCERTAIN, UNAVAILABLE
    confidence: float
    reason: str
    relationship_type: str


def should_ai_review(match_result) -> bool:
    risky_passes = {
        "address_only", "address_name_related", "address_domain", "address_secondary_or_acronym",
        "domain_name_related", "family_bridge_supported", "family_cross_country",
        "secondary_or_acronym_bridge", "acronym_review_candidate", "name_fuzzy_strong",
        "known_brand_family_alias", "support_field_review", "domain_review_candidate",
        "person_address_city_mismatch_review", "regulatory_or_task_force_related",
        "tax_exact_low_similarity_review",
    }
    return match_result.pass_type in risky_passes or (45 <= match_result.match_pct <= 80 and match_result.needs_review)


def _compact_record(row: Dict[str, Any]) -> Dict[str, Any]:
    # Include the mapped original fields plus normalized values. This is still compact,
    # but gives the LLM the full supplier name/address/domain/tax context needed for
    # ambiguous DBA, family, and parent-bridge decisions.
    return {
        "original_name": row.get("orig_supplier_name", ""),
        "original_address": row.get("orig_address", ""),
        "original_city": row.get("orig_city", ""),
        "original_country": row.get("orig_country", ""),
        "original_email": row.get("orig_email", ""),
        "original_website": row.get("orig_website", ""),
        "original_domain": row.get("orig_domain", ""),
        "original_secondary_names": row.get("orig_secondary_names", ""),
        "name_norm": row.get("name_norm", ""),
        "address_norm": row.get("addr_norm", ""),
        "city_norm": row.get("city_norm", ""),
        "country_norm": row.get("country_norm", ""),
        "tax_norm": row.get("tax_norm", ""),
        "tax_loose_norm": row.get("tax_loose_norm", ""),
        "domain": row.get("domain", ""),
        "root_brand": row.get("root_brand", ""),
    }


def _parse_json_response(text: str) -> AIReviewResult:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end]) if start >= 0 and end > start else json.loads(text)
        decision = str(data.get("decision") or data.get("verdict") or "uncertain").lower()
        if decision in {"cluster", "same", "same_entity", "related"}:
            verdict = "RELATED" if decision == "related" else "SAME_ENTITY"
        elif decision in {"do_not_cluster", "different", "different_entity", "no"}:
            verdict = "DIFFERENT"
        elif decision == "partial":
            # The current engine reviews pairs before merge. For a two-record pair,
            # a partial group response cannot safely approve a full edge unless a
            # future group-level LLM workflow maps returned row_ids back to split
            # clusters. Treat it as no pair merge by default.
            verdict = "DIFFERENT"
        else:
            verdict = "UNCERTAIN"
        pct = float(data.get("match_percentage") or data.get("confidence") or 50)
        if pct <= 1:
            pct *= 100
        return AIReviewResult(verdict, pct, str(data.get("reason", "")), str(data.get("relationship_type", "unknown")))
    except Exception as e:
        return AIReviewResult("UNAVAILABLE", 0.0, f"Could not parse AI response: {e}. Raw: {text[:300]}", "unknown")


def ai_review_pair(row_a: Dict[str, Any], row_b: Dict[str, Any], config=None) -> AIReviewResult:
    enabled = os.getenv("AI_REVIEW_ENABLED", "false").lower() == "true"
    if config is not None:
        enabled = bool(getattr(config, "ai_review_enabled", enabled))
    if not enabled:
        return AIReviewResult("UNAVAILABLE", 0.0, "AI review disabled", "unknown")

    provider = getattr(config, "ai_provider", os.getenv("LLM_PROVIDER") or os.getenv("AI_PROVIDER", "openai_compatible")) if config else (os.getenv("LLM_PROVIDER") or os.getenv("AI_PROVIDER", "openai_compatible"))
    api_key = getattr(config, "ai_api_key", "") if config else ""
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    base_url = getattr(config, "ai_base_url", os.getenv("AI_BASE_URL", "https://api.openai.com/v1")) if config else os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    model = getattr(config, "ai_model", os.getenv("LLM_MODEL") or os.getenv("AI_MODEL", "gpt-4.1-mini")) if config else (os.getenv("LLM_MODEL") or os.getenv("AI_MODEL", "gpt-4.1-mini"))
    timeout = getattr(config, "ai_timeout_seconds", 45) if config else 45
    provider_normalized = str(provider or "").lower()
    if provider_normalized in {"gemini", "google", "google_gemini"}:
        if not base_url or base_url == "https://api.openai.com/v1":
            base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        if not model or model == "gpt-4.1-mini":
            model = os.getenv("LLM_MODEL") or os.getenv("AI_MODEL") or "gemini-2.5-flash"

    if not api_key:
        return AIReviewResult("UNAVAILABLE", 0.0, "AI API key missing", "unknown")

    cache_key = _cache_key(row_a, row_b, provider_normalized, model)
    cache_enabled = bool(getattr(config, "ai_cache_enabled", True)) if config is not None else True
    cache_path = getattr(config, "ai_cache_path", ".cache/llm_review_cache.json") if config is not None else ".cache/llm_review_cache.json"
    if cache_enabled:
        cached = _cache_get(cache_path, cache_key)
        if cached:
            return AIReviewResult(**cached)

    max_calls = int(getattr(config, "ai_max_calls", 50) or 50) if config is not None else int(os.getenv("AI_MAX_CALLS", "50"))
    if config is not None:
        current_calls = int(getattr(config, "_ai_review_calls", 0))
        if current_calls >= max_calls:
            return AIReviewResult("UNAVAILABLE", 0.0, f"AI review budget exhausted ({max_calls} calls)", "unknown")
        setattr(config, "_ai_review_calls", current_calls + 1)

    prompt = f"""You are reviewing supplier/vendor clustering candidates.
Goal: parent/family-level clustering, not only exact legal entity.
Decide whether these two records should be placed in the same suggested cluster for human review.
Be conservative for same-address-only cases, people, hotels, restaurants, and franchise chains.
Different tax IDs can still be related at parent/family level, but mention uncertainty.
If the evidence is plausible but not fully proven, return decision = uncertain with a lower match_percentage (usually 65-70). The system will keep uncertain records in the separate Review output by default, not the main cluster output.
Return JSON only with keys: decision, match_percentage, clusters, exclude_row_ids, relationship_type, reason.
Allowed decision values: cluster, do_not_cluster, uncertain, partial.
For this pair-level review, use partial only if a future group-level workflow should split a larger group; pair-level partial responses are treated as no pair merge.

Record A: {json.dumps(_compact_record(row_a), ensure_ascii=False)}
Record B: {json.dumps(_compact_record(row_b), ensure_ascii=False)}
"""

    try:
        if provider_normalized == "claude":
            url = base_url.rstrip("/") + "/messages"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            payload = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]}
            r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json().get("content", [])
            text = content[0].get("text", "") if content else ""
        elif provider_normalized in {"gemini", "google", "google_gemini"}:
            url = base_url.rstrip("/") + f"/models/{model}:generateContent"
            headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            }
            r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            text = "".join(part.get("text", "") for part in parts)
        else:
            # OpenAI-compatible APIs.
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        result = _parse_json_response(text)
        if cache_enabled:
            _cache_put(cache_path, cache_key, result)
        return result
    except Exception as e:
        return AIReviewResult("UNAVAILABLE", 0.0, f"AI call failed: {e}", "unknown")


def _cache_key(row_a: Dict[str, Any], row_b: Dict[str, Any], provider: str, model: str) -> str:
    records = [_compact_record(row_a), _compact_record(row_b)]
    records = sorted(records, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    payload = {"provider": provider, "model": model, "records": records}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _cache_get(path: str, key: str) -> Optional[Dict[str, Any]]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        value = data.get(key)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _cache_put(path: str, key: str, result: AIReviewResult) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        data[key] = {
            "verdict": result.verdict,
            "confidence": result.confidence,
            "reason": result.reason,
            "relationship_type": result.relationship_type,
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return
