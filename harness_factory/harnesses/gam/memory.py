"""
GAMMemory: Generative Agent Memory with dual-agent architecture.

Faithfully reproduces the GAM (General-Agentic-Memory) framework:
  - **MemoryAgent** (every step): Generates a concise abstract for each new
    message, maintains a page store of decorated pages, and builds memory context.
  - **ResearchAgent** (every K steps): Performs a structured plan→search→
    integrate→reflect loop over accumulated pages to produce an integrated
    memory summary that serves as the reorganized context.

Architecture:
  - Abstracts list = MemoryStore  (list of abstract strings)
  - Pages list     = PageStore    (list of Page objects with header/content/meta)
  - BM25 retriever = keyword search over page content
  - FAISS retriever = dense vector search over page content
  - Page index     = direct page access by integer index

Dependencies: faiss-cpu, sentence-transformers, rank_bm25
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)





# ═══════════════════════════════════════════════════════════════════════
# DATA TYPES (mirrors GAM schemas, kept lightweight as dataclasses)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Page:
    """A single page in the page store."""
    header: str = ""
    content: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPlan:
    """Output of the PlanningAgent."""
    info_needs: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    keyword_collection: List[str] = field(default_factory=list)
    vector_queries: List[str] = field(default_factory=list)
    page_index: List[int] = field(default_factory=list)


@dataclass
class Hit:
    """A single retrieval hit."""
    page_id: str = ""
    snippet: str = ""
    source: str = ""  # "keyword" | "vector" | "page_index"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Working result for the ResearchAgent."""
    content: str = ""
    sources: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════

class MemoryStrategy(BaseMemory):
    """Generative Agent Memory with MemoryAgent + ResearchAgent.

    - At EVERY update(step): MemoryAgent generates an abstract, creates a Page.
    - Every K steps: ResearchAgent performs plan→search→integrate→reflect to
      produce an integrated_memory (the reorganized context).
    - build_context() returns: system_prompt + task + integrated_memory +
      recent raw steps.

    Configuration:
        reorg_interval: ResearchAgent reorganization every K steps.
        max_research_iters: Max iterations for the research loop.
        top_k: Number of retrieval hits per search channel.
        recent_window: Number of recent raw steps to always include.
        embedding_model: sentence-transformers model name for dense retrieval.
    """

    def __init__(
        self,
        prompts=None,
        reorg_interval: int = 4,
        max_research_iters: int = 3,
        top_k: int = 5,
        recent_window: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.prompts = prompts or {}
        self._reorg_interval = reorg_interval
        self._max_research_iters = max_research_iters
        self._top_k = top_k
        self._recent_window = recent_window
        self._embedding_model_name = embedding_model

        # Lazy-loaded
        self._encoder = None        # SentenceTransformer
        self._faiss_index = None    # faiss.IndexFlatIP
        self._bm25 = None           # BM25Okapi
        self._embedding_dim: int = 0

        # State
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._model: Optional[Callable] = None

        # MemoryAgent state
        self._abstracts: List[str] = []           # MemoryStore: list of abstract strings
        self._pages: List[Page] = []              # PageStore: list of Page objects
        self._page_embeddings: List[np.ndarray] = []  # Dense embeddings for pages

        # ResearchAgent state
        self._integrated_memory: str = ""         # Last research output
        self._research_count: int = 0

        # Step tracking
        self._all_steps: List[Union[StepRecord, PlanState, SummaryState]] = []
        self._steps_since_reorg: int = 0
        self._last_page_count_for_bm25: int = 0

    def set_model(self, model: Any) -> None:
        """Set the LLM model. Called by runtime."""
        self._model = model

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._abstracts = []
        self._pages = []
        self._page_embeddings = []
        self._integrated_memory = ""
        self._research_count = 0
        self._all_steps = []
        self._steps_since_reorg = 0
        self._last_page_count_for_bm25 = 0

        # Reset retrieval indices
        self._faiss_index = None
        self._bm25 = None

        # Lazy-load encoder
        self._ensure_encoder()

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Build context: system + task + integrated_memory + recent steps."""
        messages: List[Message] = []

        # System prompt
        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self._system_prompt}],
            )
        )

        # Task
        messages.extend(self._task.to_messages())

        # Integrated memory from ResearchAgent (if available)
        if self._integrated_memory:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{
                        "type": "text",
                        "text": (
                            "### Integrated Memory Context\n"
                            "The following is a reorganized summary of all relevant "
                            "information gathered so far:\n\n"
                            + self._integrated_memory
                        ),
                    }],
                )
            )
        elif self._abstracts:
            # Before first research, show recent abstracts as context
            abstract_text = self._format_memory_context()
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{
                        "type": "text",
                        "text": f"### Memory Context\n{abstract_text}",
                    }],
                )
            )

        # Recent raw steps (always included for immediate context)
        recent_steps = self._get_recent_steps()
        for step in recent_steps:
            messages.extend(step.to_messages())

        return MemoryView(
            messages=messages,
            metadata={
                "num_abstracts": len(self._abstracts),
                "num_pages": len(self._pages),
                "research_count": self._research_count,
                "has_integrated_memory": bool(self._integrated_memory),
            },
        )

    def update(self, step: StepRecord) -> None:
        """At EVERY step: MemoryAgent memorizes. Every K: ResearchAgent researches."""
        self._all_steps.append(step)
        self._steps_since_reorg += 1

        # ── MemoryAgent: memorize at every step ──
        message = self._format_step_as_message(step)
        self._memorize(message)

        # ── ResearchAgent: reorganize every K steps ──
        if self._steps_since_reorg >= self._reorg_interval:
            self._research()
            self._steps_since_reorg = 0

    def update_plan(self, plan: PlanState) -> None:
        self._all_steps.append(plan)
        # Memorize plan as a message
        self._memorize(f"Agent created a plan: {plan.plan}")

    def update_summary(self, summary: SummaryState) -> None:
        self._all_steps.append(summary)
        # Memorize summary as a message
        self._memorize(f"Agent generated a progress summary: {summary.summary}")

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._all_steps if isinstance(s, StepRecord)]

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY AGENT (called at every step)
    # ═══════════════════════════════════════════════════════════════════

    def _memorize(self, message: str) -> None:
        """MemoryAgent.memorize(): generate abstract, create Page, store both.

        Mirrors original memory_agent.py memorize() method.
        """
        message = message.strip()
        if not message:
            return

        # Build memory context from all existing abstracts
        # Mirrors original: "Page 0: ..., Page 1: ..., etc."
        if self._abstracts:
            memory_context_lines = []
            for i, abstract in enumerate(self._abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        else:
            memory_context = "No memory currently."

        # Generate abstract via LLM
        abstract = self._generate_abstract(message, memory_context)

        # Add abstract to memory store (with uniqueness check)
        if abstract and abstract not in self._abstracts:
            self._abstracts.append(abstract)

        # Create and store Page
        header = f"[ABSTRACT] {abstract}".strip()
        decorated_page = f"{header}; {message}"
        page = Page(header=header, content=message, meta={"decorated": decorated_page})
        self._pages.append(page)

        # Encode page content for dense retrieval
        embedding = self._encode(page.content)
        if embedding is not None:
            self._page_embeddings.append(embedding)
            # Update FAISS index
            if self._faiss_index is not None:
                self._faiss_index.add(embedding.reshape(1, -1))

    def _generate_abstract(self, message: str, memory_context: str) -> str:
        """Generate abstract using MEMORY_AGENT_PROMPT."""
        if self._model is None:
            return message[:200]

        prompt = self.prompts["memory"]["memory_agent_prompt"].format(
            input_message=message,
            memory_context=memory_context,
        )

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

        try:
            response = self._model(messages)
            text = (response.content or "").strip()
            # Strip <think> tags if present
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text if text else message[:200]
        except Exception as e:
            logger.error(f"GAMMemory: abstract generation failed: {e}")
            return message[:200]

    # ═══════════════════════════════════════════════════════════════════
    # RESEARCH AGENT (called every K steps)
    # ═══════════════════════════════════════════════════════════════════

    def _research(self) -> None:
        """ResearchAgent.research(): plan→search→integrate→reflect loop.

        Mirrors original research_agent.py research() method.
        """
        if self._model is None:
            logger.warning("GAMMemory: no model, skipping research")
            return

        task = self._task.task if self._task else ""
        if not task:
            return

        # Update retrieval indices before research
        self._update_retrievers()

        temp = Result()
        next_request = task

        for step in range(self._max_research_iters):
            # Planning
            plan = self._planning(next_request)

            # Search + Integrate
            temp = self._search_and_integrate(plan, temp, task)

            # Reflection
            enough, new_request = self._reflection(task, temp)

            if enough:
                break

            next_request = new_request if new_request else task

        # Store the integrated result
        if temp.content:
            self._integrated_memory = temp.content
            self._research_count += 1
            logger.info(
                f"GAMMemory: research #{self._research_count} complete "
                f"(len={len(temp.content)})"
            )

    def _planning(self, request: str) -> SearchPlan:
        """PlanningAgent: generate a structured retrieval plan."""
        memory_context = self._format_memory_context()

        prompt = self.prompts["memory"]["planning_prompt"].format(
            request=request, memory=memory_context
        )

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

        try:
            response = self._model(messages)
            text = (response.content or "").strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            # Extract JSON from response
            data = self._parse_json(text)
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", []),
            )
        except Exception as e:
            logger.error(f"GAMMemory: planning failed: {e}")
            return SearchPlan()

    def _search_and_integrate(
        self, plan: SearchPlan, result: Result, question: str,
    ) -> Result:
        """Execute search plan and integrate results via LLM."""
        all_hits: List[Hit] = []

        for tool in plan.tools:
            if tool == "keyword" and plan.keyword_collection:
                hits = self._search_by_keyword(plan.keyword_collection)
                all_hits.extend(hits)
            elif tool == "vector" and plan.vector_queries:
                hits = self._search_by_vector(plan.vector_queries)
                all_hits.extend(hits)
            elif tool == "page_index" and plan.page_index:
                hits = self._search_by_page_index(plan.page_index)
                all_hits.extend(hits)

        if not all_hits:
            return result

        # Deduplicate by page_id (keep highest score)
        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:
                existing = unique_hits.get(hit.page_id)
                if existing is None:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_score = existing.meta.get("score", 0)
                    current_score = hit.meta.get("score", 0)
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)

        sorted_hits = sorted(
            list(unique_hits.values()) + hits_without_id,
            key=lambda h: h.meta.get("score", 0),
            reverse=True,
        )

        # Integrate via LLM
        return self._integrate(sorted_hits, result, question)

    def _integrate(self, hits: List[Hit], result: Result, question: str) -> Result:
        """IntegrateAgent: merge search evidence with current result."""
        evidence_parts = []
        sources = []
        for i, hit in enumerate(hits, 1):
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
                sources.append(hit.page_id)
            evidence_parts.append(f"{i}. {source_info} {hit.snippet}")

        evidence_context = "\n".join(evidence_parts) if evidence_parts else "No search results"

        prompt = self.prompts["memory"]["integrate_prompt"].format(
            question=question,
            evidence_context=evidence_context,
            result=result.content,
        )

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

        try:
            response = self._model(messages)
            text = (response.content or "").strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            data = self._parse_json(text)
            return Result(
                content=data.get("content", ""),
                sources=[str(s) for s in data.get("sources", sources) if s],
            )
        except Exception as e:
            logger.error(f"GAMMemory: integration failed: {e}")
            return result

    def _reflection(self, request: str, result: Result) -> Tuple[bool, Optional[str]]:
        """Two-step reflection: InfoCheck + GenerateRequests."""
        # Step 1: Check completeness
        check_prompt = self.prompts["memory"]["info_check_prompt"].format(
            request=request, result=result.content
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": check_prompt}]}]

        try:
            response = self._model(messages)
            text = (response.content or "").strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            data = self._parse_json(text)
            enough = data.get("enough", False)

            if enough:
                return True, None

            # Step 2: Generate follow-up requests
            gen_prompt = self.prompts["memory"]["generate_requests_prompt"].format(
                request=request, result=result.content,
            )
            gen_msgs = [{"role": "user", "content": [{"type": "text", "text": gen_prompt}]}]

            response = self._model(gen_msgs)
            text = (response.content or "").strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            data = self._parse_json(text)
            new_requests = data.get("new_requests", [])
            new_request = " ".join(new_requests) if new_requests else None
            return False, new_request

        except Exception as e:
            logger.error(f"GAMMemory: reflection failed: {e}")
            return False, None

    # ═══════════════════════════════════════════════════════════════════
    # SEARCH CHANNELS
    # ═══════════════════════════════════════════════════════════════════

    def _search_by_keyword(self, queries: List[str]) -> List[Hit]:
        """BM25 keyword search over page content."""
        if self._bm25 is None or not self._pages:
            # Fallback: substring search
            return self._fallback_keyword_search(queries)

        hits = []
        combined = " ".join(queries)
        tokenized_query = combined.lower().split()

        try:
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:self._top_k]
            for idx in top_indices:
                if scores[idx] > 0:
                    hits.append(Hit(
                        page_id=str(idx),
                        snippet=self._pages[idx].content[:500],
                        source="keyword",
                        meta={"score": float(scores[idx])},
                    ))
        except Exception as e:
            logger.error(f"GAMMemory: BM25 search failed: {e}")
            return self._fallback_keyword_search(queries)

        return hits

    def _fallback_keyword_search(self, queries: List[str]) -> List[Hit]:
        """Naive substring search fallback."""
        hits = []
        for query in queries:
            q = query.lower()
            for i, page in enumerate(self._pages):
                if q in page.content.lower() or q in page.header.lower():
                    hits.append(Hit(
                        page_id=str(i),
                        snippet=page.content[:500],
                        source="keyword",
                        meta={},
                    ))
                    if len(hits) >= self._top_k:
                        return hits
        return hits

    def _search_by_vector(self, queries: List[str]) -> List[Hit]:
        """Dense vector search via FAISS."""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        hits = []
        # Multi-query: aggregate scores by page_id across all queries
        score_map: Dict[int, float] = {}
        for query in queries:
            embedding = self._encode(query)
            if embedding is None:
                continue

            k = min(self._top_k, self._faiss_index.ntotal)
            try:
                scores, indices = self._faiss_index.search(
                    embedding.reshape(1, -1), k
                )
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self._pages):
                        score_map[idx] = score_map.get(idx, 0) + float(score)
            except Exception as e:
                logger.error(f"GAMMemory: FAISS search failed: {e}")

        # Sort by aggregated score
        sorted_pages = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        for idx, score in sorted_pages[:self._top_k]:
            hits.append(Hit(
                page_id=str(idx),
                snippet=self._pages[idx].content[:500],
                source="vector",
                meta={"score": score},
            ))

        return hits

    def _search_by_page_index(self, page_indices: List[int]) -> List[Hit]:
        """Direct page access by index."""
        hits = []
        for idx in page_indices:
            if 0 <= idx < len(self._pages):
                hits.append(Hit(
                    page_id=str(idx),
                    snippet=self._pages[idx].content,
                    source="page_index",
                    meta={},
                ))
        return hits

    # ═══════════════════════════════════════════════════════════════════
    # RETRIEVER MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _update_retrievers(self) -> None:
        """Rebuild BM25 index if page count changed."""
        current_count = len(self._pages)
        if current_count != self._last_page_count_for_bm25 and current_count > 0:
            self._rebuild_bm25()
            self._last_page_count_for_bm25 = current_count

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all pages."""
        try:
            from rank_bm25 import BM25Okapi

            corpus = [page.content.lower().split() for page in self._pages]
            self._bm25 = BM25Okapi(corpus)
            logger.debug(f"GAMMemory: rebuilt BM25 index ({len(corpus)} pages)")
        except ImportError:
            logger.warning("GAMMemory: rank_bm25 not installed, keyword search uses fallback")
        except Exception as e:
            logger.error(f"GAMMemory: BM25 rebuild failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # ENCODING & FAISS
    # ═══════════════════════════════════════════════════════════════════

    def _ensure_encoder(self) -> None:
        """Lazy-load sentence-transformers and FAISS."""
        if self._encoder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            self._encoder = SentenceTransformer(self._embedding_model_name)
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()
            self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)

            logger.info(
                f"GAMMemory: loaded encoder '{self._embedding_model_name}' "
                f"(dim={self._embedding_dim}), FAISS index ready"
            )
        except ImportError as e:
            logger.error(
                f"GAMMemory: missing dependency: {e}. "
                "Install with: pip install faiss-cpu sentence-transformers"
            )
            self._encoder = None
            self._faiss_index = None

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text to a normalized dense vector."""
        if self._encoder is None:
            return None

        try:
            embedding = self._encoder.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"GAMMemory: encoding failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _format_step_as_message(self, step: StepRecord) -> str:
        """Format a StepRecord into the raw message text for MemoryAgent."""
        parts = []
        if step.action_think:
            parts.append(f"Agent thought: {step.action_think}")
        if step.tool_calls:
            for tc in step.tool_calls:
                parts.append(f"Tool call: {tc.name}({tc.arguments})")
        if step.observations:
            parts.append(f"Tool response: {step.observations}")
        if step.error:
            parts.append(f"Error: {step.error}")
        return "\n".join(parts) if parts else "Empty step"

    def _format_memory_context(self) -> str:
        """Format all abstracts as 'Page 0: ..., Page 1: ...'."""
        if not self._abstracts:
            return "No memory currently."
        lines = []
        for i, abstract in enumerate(self._abstracts):
            lines.append(f"Page {i}: {abstract}")
        return "\n".join(lines)

    def _get_recent_steps(self) -> List[Union[StepRecord, PlanState, SummaryState]]:
        """Get the most recent raw action steps."""
        if not self._all_steps:
            return []
        recent = []
        count = 0
        for step in reversed(self._all_steps):
            if isinstance(step, StepRecord):
                recent.append(step)
                count += 1
                if count >= self._recent_window:
                    break
        recent.reverse()
        return recent

    def _parse_json(self, text: str) -> dict:
        """Robustly extract JSON from LLM output."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting JSON from code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except (json.JSONDecodeError, TypeError, IndexError):
                    continue

        # Try json_repair as last resort
        try:
            import json_repair
            return json_repair.loads(text)
        except Exception:
            pass

        return {}
