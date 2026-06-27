"""LLM answer generation with grounded medical citations.

Responsibilities
----------------
- Provider abstraction: OpenAI-compatible (Ollama, vLLM, etc.) and Anthropic
- Medical system prompt that enforces citation discipline
- Extract [N] citations from generated answers and resolve to chunk metadata
- Hallucination heuristic: score each answer sentence against retrieved context
- Streaming support
- Structured output with Citation + GenerationResult dataclasses
"""
from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from .config import LLMConfig
from .context_assembler import AssembledContext, ContextChunk


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    number: int                     # [N] citation number as appears in the answer
    chunk_id: int
    page_start: int
    page_end: int
    section: str
    quote: str                      # verbatim supporting text from the chunk
    confidence: float               # lexical overlap score (0–1)
    source_label: str


@dataclass
class GenerationResult:
    query: str
    answer: str
    citations: List[Citation]
    faithfulness_score: float       # fraction of answer sentences with ≥1 supporting citation
    tokens_used: Dict[str, int]     # prompt_tokens, completion_tokens, total_tokens
    latency_ms: float
    model_name: str
    provider: str
    context_tokens: int
    truncated_context: bool


# ─────────────────────────────────────────────────────────────────────────────
# Medical system prompt
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """You are a clinical evidence assistant. Your task is to answer questions about a patient's medical record using ONLY the provided evidence passages.

RULES:
1. Answer based exclusively on the provided evidence. Do not use general medical knowledge to fill in gaps.
2. Cite the evidence for every specific claim using [N] notation, where N is the evidence number shown above each passage.
3. If multiple passages support the same claim, cite all: [1][3].
4. If the evidence is insufficient, state: "The provided documents do not contain sufficient information to answer this question."
5. For medications, always state the exact dosage and frequency as written in the evidence.
6. For diagnoses, preserve the qualifier as stated (e.g., "probable", "possible", "confirmed", "ruled out").
7. Never speculate, infer, or extrapolate beyond what is explicitly in the evidence.
8. Do not apologise or add filler phrases. Be direct and precise.

FORMAT:
- Begin with a direct, complete answer.
- Follow with details, each supported by inline citations.
- Keep the answer concise. Every sentence should be supported by evidence."""


# ─────────────────────────────────────────────────────────────────────────────
# Provider ABC
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate a completion. Returns full string or token iterator."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible provider (Ollama, vLLM, LMStudio, OpenAI, Groq…)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleProvider(LLMProvider):
    """OpenAI SDK — works with any OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai is required. Install with: uv pip install openai") from e

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": config.timeout_seconds}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self._client = openai.OpenAI(**kwargs)
        self._config = config

    def complete(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        kwargs: Dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": stream,
        }
        if stream:
            return self._stream_response(self._client.chat.completions.create(**kwargs))

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    @staticmethod
    def _stream_response(stream) -> Iterator[str]:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete_with_usage(
        self,
        messages: List[Dict[str, str]],
    ) -> tuple[str, Dict[str, int]]:
        """Non-streaming completion that also returns token usage."""
        resp = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stream=False,
        )
        text = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if resp.usage:
            usage = {
                "prompt_tokens": int(resp.usage.prompt_tokens or 0),
                "completion_tokens": int(resp.usage.completion_tokens or 0),
                "total_tokens": int(resp.usage.total_tokens or 0),
            }
        return text, usage

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def provider_name(self) -> str:
        if self._config.base_url:
            host = re.sub(r"https?://|/.*$", "", self._config.base_url)
            return f"openai_compatible({host})"
        return "openai"


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic provider
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider. Install: uv pip install 'nitrag[anthropic]'"""

    def __init__(self, config: LLMConfig) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic is required. Install with: uv pip install 'nitrag[anthropic]'"
            ) from e

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._config = config

    def complete(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        if stream:
            return self._stream_anthropic(system_msg, user_messages)

        resp = self._client.messages.create(
            model=self._config.model_name,
            system=system_msg,
            messages=user_messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        return resp.content[0].text if resp.content else ""

    def _stream_anthropic(self, system_msg: str, messages: list) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._config.model_name,
            system=system_msg,
            messages=messages,
            max_tokens=self._config.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def complete_with_usage(self, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, int]]:
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        resp = self._client.messages.create(
            model=self._config.model_name,
            system=system_msg,
            messages=user_messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        text = resp.content[0].text if resp.content else ""
        usage = {
            "prompt_tokens": int(resp.usage.input_tokens),
            "completion_tokens": int(resp.usage.output_tokens),
            "total_tokens": int(resp.usage.input_tokens + resp.usage.output_tokens),
        }
        return text, usage

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def provider_name(self) -> str:
        return "anthropic"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "openai_compatible":
        return OpenAICompatibleProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    raise ValueError(
        f"Unknown LLM provider: {config.provider!r}. "
        "Options: 'openai_compatible', 'anthropic'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Citation extraction
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citation_numbers(text: str) -> List[int]:
    """Extract all [N] citation numbers from an answer text."""
    return [int(m.group(1)) for m in _CITATION_RE.finditer(text)]


def resolve_citations(
    answer: str,
    context: AssembledContext,
    min_confidence: float = 0.15,
) -> List[Citation]:
    """Map [N] citation markers in the answer to ContextChunk metadata.

    For each [N], we find the answer sentences that explicitly reference it,
    then match those specific sentences against the chunk text.  This gives a
    precise source quote that the render_page endpoint can highlight exactly.
    """
    chunk_by_citation: Dict[int, ContextChunk] = {
        c.citation_number: c for c in context.chunks
    }

    # Build: citation_number → list of answer sentences that contain [N]
    answer_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    citation_to_sents: Dict[int, List[str]] = {}
    for sent in answer_sents:
        for n in {int(m) for m in _CITATION_RE.findall(sent)}:
            citation_to_sents.setdefault(n, []).append(sent)

    seen: set = set()
    citations: List[Citation] = []

    for n in sorted(set(extract_citation_numbers(answer))):
        if n in seen or n not in chunk_by_citation:
            continue
        seen.add(n)
        c = chunk_by_citation[n]

        # Sentence-level chunk: the chunk text IS the source sentence — no F1 needed
        if c.chunk_strategy_name == "sentence_based":
            citations.append(Citation(
                number=n,
                chunk_id=c.chunk_id,
                page_start=c.page_start,
                page_end=c.page_end,
                section=c.section,
                quote=c.text.strip(),
                confidence=1.0,
                source_label=c.source_label,
            ))
            continue

        # Match each answer sentence that references [N] against the chunk
        ans_sents_for_n = citation_to_sents.get(n, [])
        best_quote = ""
        best_confidence = 0.0

        for ans_sent in ans_sents_for_n:
            quote, score = _best_matching_chunk_sentence(ans_sent, c.text)
            if score > best_confidence and quote:
                best_quote = quote
                best_confidence = score

        # Fallback: use the first 300 chars of the chunk
        if not best_quote or best_confidence < min_confidence:
            best_quote = c.text[:300].strip()
            best_confidence = 0.0

        citations.append(Citation(
            number=n,
            chunk_id=c.chunk_id,
            page_start=c.page_start,
            page_end=c.page_end,
            section=c.section,
            quote=best_quote,
            confidence=round(best_confidence, 3),
            source_label=c.source_label,
        ))

    return sorted(citations, key=lambda x: x.number)


_QUOTE_STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "by",
    "for", "with", "and", "or", "but", "from", "that", "this", "it", "as",
    "no", "yes", "also", "about", "patient", "patients",
})


def _content_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _QUOTE_STOPWORDS


def _f1_score(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    prec = inter / len(b_tokens)
    rec  = inter / len(a_tokens)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def _best_matching_chunk_sentence(answer_sentence: str, chunk_text: str) -> tuple[str, float]:
    """Find the sentence(s) in chunk_text most similar to a specific answer sentence.

    Uses F1 on content tokens (stopwords removed) so short answer sentences don't
    dominate recall unfairly.  Returns up to two adjacent high-scoring sentences
    joined together when both score well, to give the renderer enough text to find.
    """
    a_tok = _content_tokens(answer_sentence)
    if not a_tok:
        return chunk_text[:300].strip(), 0.0

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text.strip()) if len(s.strip()) >= 8]
    if not sents:
        return chunk_text[:300].strip(), 0.0

    scored = []
    for sent in sents:
        s_tok = _content_tokens(sent)
        scored.append((sent, _f1_score(a_tok, s_tok)))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_sent, best_score = scored[0]

    # If the second sentence also scores ≥ 60% of the best, include it
    # (supports multi-sentence evidence for a single claim)
    result_sents = [best_sent]
    if len(scored) > 1:
        second_sent, second_score = scored[1]
        if second_score >= best_score * 0.60 and second_score >= 0.15:
            result_sents.append(second_sent)

    return " ".join(result_sents)[:500], best_score


def _best_supporting_quote(answer: str, chunk_text: str) -> tuple[str, float]:
    """Legacy wrapper — kept for any external callers; prefers sentence-level matching."""
    return _best_matching_chunk_sentence(answer, chunk_text)


def build_sentence_citations(
    answer: str,
    context: AssembledContext,
    min_score: float = 0.12,
) -> List[Dict[str, Any]]:
    """Post-generation step: map each answer sentence to its source chunk sentences.

    Returns a list of:
    {
        "sentence_idx": int,
        "sentence": str,
        "citations": [{"citation_number", "chunk_id", "page_start", "page_end",
                       "quote", "score", "section", "source_label", "inferred"}]
    }
    """
    chunk_by_n: Dict[int, ContextChunk] = {c.citation_number: c for c in context.chunks}

    answer_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]

    result = []
    for idx, sent in enumerate(answer_sents):
        cited_ns = sorted({int(m) for m in _CITATION_RE.findall(sent)})
        sent_citations = []

        # Explicit [N] citations in this sentence → find exact source sentence
        for n in cited_ns:
            if n not in chunk_by_n:
                continue
            c = chunk_by_n[n]

            # Sentence-level chunk: use full chunk text directly
            if c.chunk_strategy_name == "sentence_based":
                sent_citations.append({
                    "citation_number": n,
                    "chunk_id":        c.chunk_id,
                    "page_start":      c.page_start,
                    "page_end":        c.page_end,
                    "quote":           c.text.strip(),
                    "score":           1.0,
                    "section":         c.section,
                    "source_label":    c.source_label,
                    "inferred":        False,
                })
                continue

            quote, score = _best_matching_chunk_sentence(sent, c.text)
            if not quote:
                quote = c.text[:300].strip()
            sent_citations.append({
                "citation_number": n,
                "chunk_id":        c.chunk_id,
                "page_start":      c.page_start,
                "page_end":        c.page_end,
                "quote":           quote,
                "score":           round(score, 3),
                "section":         c.section,
                "source_label":    c.source_label,
                "inferred":        False,
            })

        # No explicit citation — search all chunks for the best match
        if not cited_ns:
            candidates = []
            for c in context.chunks:
                quote, score = _best_matching_chunk_sentence(sent, c.text)
                if score >= min_score and quote:
                    candidates.append({
                        "citation_number": c.citation_number,
                        "chunk_id":        c.chunk_id,
                        "page_start":      c.page_start,
                        "page_end":        c.page_end,
                        "quote":           quote,
                        "score":           round(score, 3),
                        "section":         c.section,
                        "source_label":    c.source_label,
                        "inferred":        True,
                    })
            candidates.sort(key=lambda x: x["score"], reverse=True)
            sent_citations = candidates[:2]

        result.append({
            "sentence_idx": idx,
            "sentence":     sent,
            "citations":    sent_citations,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness heuristic
# ─────────────────────────────────────────────────────────────────────────────

def compute_faithfulness(answer: str, context: AssembledContext) -> float:
    """Fraction of answer sentences that have ≥1 lexical overlap with cited chunks.

    This is a lightweight heuristic — not a semantic entailment check.
    Sentences with citation markers [N] are considered faithful by default.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        return 1.0

    context_text = " ".join(c.text for c in context.chunks).lower()
    context_tokens = set(re.findall(r"[a-z0-9]+", context_text))

    supported = 0
    for sent in sentences:
        # If sentence has a citation marker, treat as supported
        if _CITATION_RE.search(sent):
            supported += 1
            continue
        sent_tokens = set(re.findall(r"[a-z0-9]+", sent.lower()))
        if len(sent_tokens) == 0:
            continue
        overlap_ratio = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap_ratio >= 0.4:
            supported += 1

    return round(supported / max(len(sentences), 1), 3)


# ─────────────────────────────────────────────────────────────────────────────
# GenerationManager
# ─────────────────────────────────────────────────────────────────────────────

class GenerationManager:
    """Generate grounded, citation-annotated answers from assembled context.

    Usage
    -----
    from nitrag.generation_manager import GenerationManager
    from nitrag.config import RAGConfig

    config = RAGConfig.local_ollama()
    gen = GenerationManager(config.llm, config.generation)

    result = gen.answer(query="What medications were prescribed?", context=assembled_context)
    print(result.answer)
    for c in result.citations:
        print(f"[{c.number}] Page {c.page_start+1} | {c.section}: {c.quote}")
    """

    def __init__(self, config: LLMConfig, generation_config=None) -> None:
        self.config = config
        self.generation_config = generation_config
        self._provider: Optional[LLMProvider] = None

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_llm_provider(self.config)
        return self._provider

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        context: AssembledContext,
        stream: bool = False,
        min_citation_confidence: float = 0.25,
    ) -> Union[GenerationResult, Iterator[str]]:
        """Generate a medical answer grounded in the assembled context.

        Set stream=True to get a token iterator instead of a GenerationResult.
        Streaming skips citation extraction and faithfulness scoring.
        """
        messages = self._build_messages(query, context)

        if stream:
            return self.provider.complete(messages, stream=True)

        t0 = time.time()
        if isinstance(self.provider, (OpenAICompatibleProvider, AnthropicProvider)):
            answer_text, usage = self.provider.complete_with_usage(messages)
        else:
            answer_text = str(self.provider.complete(messages, stream=False))
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        elapsed_ms = (time.time() - t0) * 1000

        citations = resolve_citations(answer_text, context, min_confidence=min_citation_confidence)

        faithfulness = compute_faithfulness(answer_text, context)

        return GenerationResult(
            query=query,
            answer=answer_text,
            citations=citations,
            faithfulness_score=faithfulness,
            tokens_used=usage,
            latency_ms=round(elapsed_ms, 1),
            model_name=self.provider.model_name,
            provider=self.provider.provider_name,
            context_tokens=context.total_tokens,
            truncated_context=context.truncated,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self, query: str, context: AssembledContext) -> List[Dict[str, str]]:
        system_content = (
            self.config.system_prompt
            if self.config.system_prompt
            else MEDICAL_SYSTEM_PROMPT
        )
        user_content = (
            f"EVIDENCE:\n\n{context.formatted_text}"
            f"\n\n{'─' * 60}\n\nQUESTION: {query}"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Convenience: answer without pre-assembled context
    # ------------------------------------------------------------------

    def answer_from_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        store,
        max_tokens: int = 3500,
        min_citation_confidence: float = 0.25,
    ) -> GenerationResult:
        """Convenience: assemble context and generate in one call."""
        from .context_assembler import ContextAssembler
        from .config import GenerationConfig
        gc = self.generation_config or GenerationConfig(max_context_tokens=max_tokens)
        assembler = ContextAssembler(store, gc)
        context = assembler.assemble(chunks, query, max_tokens=max_tokens)
        return self.answer(query, context, min_citation_confidence=min_citation_confidence)

    # ------------------------------------------------------------------
    # Plain-text LLM calls (no RAG context, no citation tracking)
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str) -> str:
        """Call the LLM with a single user prompt; return raw text."""
        messages = [{"role": "user", "content": prompt}]
        try:
            if isinstance(self.provider, (OpenAICompatibleProvider, AnthropicProvider)):
                text, _ = self.provider.complete_with_usage(messages)
            else:
                text = str(self.provider.complete(messages, stream=False))
            return text.strip()
        except Exception:
            return ""

    def generate_text_with_context(
        self,
        prompt: str,
        context: "AssembledContext",
    ) -> str:
        """Call the LLM with an existing assembled context injected as evidence.

        The prompt should reference the evidence; citations in the response will
        use the same [N] numbering already assigned in the context.
        """
        user_content = (
            f"EVIDENCE:\n\n{context.formatted_text}"
            f"\n\n{'─' * 60}\n\n{prompt}"
        )
        messages = [
            {"role": "system", "content": self.config.system_prompt or MEDICAL_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        try:
            if isinstance(self.provider, (OpenAICompatibleProvider, AnthropicProvider)):
                text, _ = self.provider.complete_with_usage(messages)
            else:
                text = str(self.provider.complete(messages, stream=False))
            return text.strip()
        except Exception:
            return ""
