"""Unit tests for the OpenAI / Ollama / offline LLM routing logic.

Run from the project folder:

    ../venv/Scripts/python.exe -m pytest -q

The network-calling functions (`_openai_chat`, `_ollama_chat`) and the key
resolver (`get_openai_api_key`) are monkeypatched, so no real API key, OpenAI
package, or Ollama server is needed — only the routing/fallback logic is tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import llm_helper as llm  # noqa: E402


CTX = {
    "drug_name": "TestDrug",
    "reviews": 100,
    "avg_rating": 3.2,
    "risk_ratio": 0.4,
    "keywords": ["nausea", "pain"],
    "examples": ["bad reaction after a few days"],
}


def _patch(monkeypatch, *, key, openai_ret, ollama_ret):
    """Replace the I/O boundary with counting fakes; return the call counter."""
    calls = {"openai": 0, "ollama": 0}

    monkeypatch.setattr(llm, "get_openai_api_key", lambda: key)

    def fake_openai(prompt, system, model, api_key):
        calls["openai"] += 1
        return openai_ret

    def fake_ollama(prompt, system, model):
        calls["ollama"] += 1
        return ollama_ret

    monkeypatch.setattr(llm, "_openai_chat", fake_openai)
    monkeypatch.setattr(llm, "_ollama_chat", fake_ollama)
    return calls


# --- route_chat -----------------------------------------------------------

def test_offline_calls_no_llm(monkeypatch):
    calls = _patch(monkeypatch, key="sk-x", openai_ret="OAI", ollama_ret="OLL")
    assert llm.route_chat("p", llm.PROVIDER_OFFLINE) is None
    assert calls == {"openai": 0, "ollama": 0}


def test_openai_selected_and_available_uses_openai(monkeypatch):
    calls = _patch(monkeypatch, key="sk-x", openai_ret="OAI", ollama_ret="OLL")
    assert llm.route_chat("p", llm.PROVIDER_OPENAI) == "OAI"
    assert calls["openai"] == 1
    assert calls["ollama"] == 0  # ollama not touched when openai succeeds


def test_openai_failure_falls_back_to_ollama(monkeypatch):
    calls = _patch(monkeypatch, key="sk-x", openai_ret=None, ollama_ret="OLL")
    assert llm.route_chat("p", llm.PROVIDER_OPENAI) == "OLL"
    assert calls["openai"] == 1
    assert calls["ollama"] == 1


def test_openai_selected_without_key_skips_to_ollama(monkeypatch):
    calls = _patch(monkeypatch, key=None, openai_ret="OAI", ollama_ret="OLL")
    assert llm.route_chat("p", llm.PROVIDER_OPENAI) == "OLL"
    assert calls["openai"] == 0  # no key -> never called
    assert calls["ollama"] == 1


def test_explicit_api_key_overrides_resolver(monkeypatch):
    # resolver says no key, but caller passes one explicitly -> openai used
    calls = _patch(monkeypatch, key=None, openai_ret="OAI", ollama_ret="OLL")
    assert llm.route_chat("p", llm.PROVIDER_OPENAI, api_key="sk-explicit") == "OAI"
    assert calls["openai"] == 1
    assert calls["ollama"] == 0


def test_ollama_first_succeeds_without_touching_openai(monkeypatch):
    # Ollama-first default: Ollama succeeds -> OpenAI backup never used.
    calls = _patch(monkeypatch, key="sk-x", openai_ret="OAI", ollama_ret="OLL")
    assert llm.route_chat("p", llm.PROVIDER_OLLAMA) == "OLL"
    assert calls["ollama"] == 1
    assert calls["openai"] == 0


def test_ollama_failure_falls_back_to_openai(monkeypatch):
    # New Ollama-first chain: Ollama fails -> OpenAI backup (key present).
    calls = _patch(monkeypatch, key="sk-x", openai_ret="OAI", ollama_ret=None)
    assert llm.route_chat("p", llm.PROVIDER_OLLAMA) == "OAI"
    assert calls["ollama"] == 1  # tried first
    assert calls["openai"] == 1  # backup used


def test_ollama_fails_and_no_key_returns_none(monkeypatch):
    # Ollama fails, no OpenAI key -> None (caller drops to rule-based).
    calls = _patch(monkeypatch, key=None, openai_ret="OAI", ollama_ret=None)
    assert llm.route_chat("p", llm.PROVIDER_OLLAMA) is None
    assert calls["ollama"] == 1
    assert calls["openai"] == 0  # skipped: no key


def test_all_paths_fail_returns_none(monkeypatch):
    calls = _patch(monkeypatch, key="sk-x", openai_ret=None, ollama_ret=None)
    assert llm.route_chat("p", llm.PROVIDER_OLLAMA) is None
    assert calls["ollama"] == 1
    assert calls["openai"] == 1


# --- answer_drug_question (chatbot end-to-end routing) --------------------

def test_answer_offline_is_rule_based(monkeypatch):
    calls = _patch(monkeypatch, key="sk-x", openai_ret="LLM", ollama_ret="LLM")
    out = llm.answer_drug_question("부작용은?", CTX, provider=llm.PROVIDER_OFFLINE)
    assert "TestDrug" in out  # grounded rule-based answer
    assert calls == {"openai": 0, "ollama": 0}


def test_answer_openai_returns_llm_text(monkeypatch):
    _patch(monkeypatch, key="sk-x", openai_ret="LLM_ANSWER", ollama_ret="OLL")
    out = llm.answer_drug_question("부작용은?", CTX, provider=llm.PROVIDER_OPENAI)
    assert out == "LLM_ANSWER"


def test_answer_ollama_first_default(monkeypatch):
    # Default chatbot provider is Ollama-first: Ollama answer wins.
    calls = _patch(monkeypatch, key="sk-x", openai_ret="OAI", ollama_ret="OLLAMA_ANSWER")
    out = llm.answer_drug_question("부작용은?", CTX, provider=llm.PROVIDER_OLLAMA)
    assert out == "OLLAMA_ANSWER"
    assert calls["openai"] == 0


def test_answer_falls_back_to_rule_when_llm_unavailable(monkeypatch):
    calls = _patch(monkeypatch, key="sk-x", openai_ret=None, ollama_ret=None)
    out = llm.answer_drug_question("주의할 점은?", CTX, provider=llm.PROVIDER_OPENAI)
    assert "TestDrug" in out  # rule-based fallback kicked in
    assert calls["openai"] == 1 and calls["ollama"] == 1


def test_answer_empty_context_is_handled(monkeypatch):
    _patch(monkeypatch, key=None, openai_ret=None, ollama_ret=None)
    empty_ctx = {"drug_name": "X", "reviews": 0, "avg_rating": None,
                 "risk_ratio": None, "keywords": [], "examples": []}
    out = llm.answer_drug_question("부작용?", empty_ctx, provider=llm.PROVIDER_OFFLINE)
    assert "데이터가 충분하지 않" in out


# --- get_openai_api_key ---------------------------------------------------

def test_placeholder_key_resolves_to_none(monkeypatch):
    monkeypatch.setattr(llm, "_load_env_once", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", llm._API_KEY_PLACEHOLDER)
    assert llm.get_openai_api_key() is None


def test_missing_key_resolves_to_none(monkeypatch):
    monkeypatch.setattr(llm, "_load_env_once", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.get_openai_api_key() is None


def test_real_key_is_returned(monkeypatch):
    monkeypatch.setattr(llm, "_load_env_once", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-test-key")
    assert llm.get_openai_api_key() == "sk-real-test-key"


def test_openai_available_reflects_key(monkeypatch):
    monkeypatch.setattr(llm, "get_openai_api_key", lambda: "sk-x")
    assert llm.openai_available() is True
    monkeypatch.setattr(llm, "get_openai_api_key", lambda: None)
    assert llm.openai_available() is False
