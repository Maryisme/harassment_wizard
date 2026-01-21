# llm_pipeline.py
"""
Retrieval + LLM call pipeline (fixed).

Fixes included:
- Attach the SAME OpenAIEmbeddingFunction (same EMBED_MODEL) when querying.
- Normalize source display to basename (no folder paths).
- Deduplicate citations by source (one [A n] or [B n] per unique document).
- Deterministically render Sources section from retrieval metadata:
  - Ensures [B n] items appear in Sources
  - Prevents duplicates (e.g., [A 1], [A 2], [A 3] for same doc)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or in a .env file.")

PERSIST_DIR = "data/data/chromadb"

COL_CASES = "cases"
COL_LEGIS = "legislation"

# Must match your ingestion embed model
EMBED_MODEL = "text-embedding-3-large"

# LLM
CHAT_MODEL = "gpt-4o-mini"

# Retrieval
TOP_K_CASES = 3
TOP_K_LEGIS = 3


# ----------------------------
# Prompt builder
# ----------------------------
def _build_prompt(user_situation: str, A_block: str, B_block: str, has_cases: bool) -> Dict[str, str]:
    """Constructs system and user messages cleanly."""
    system = (
        "You are a BC workplace triage explainer. "
        "Use legislation/policy excerpts to assess whether the described conduct likely qualifies as bullying or harassment. "
        "Base your analysis ONLY on the Legislation Excerpts (Section A). "
        "If NO case excerpts are provided, OMIT the 'Similar Tribunal Decisions' section entirely. "
        "If case excerpts are provided, you may include a brief 'Similar Tribunal Decisions' section using ONLY those case excerpts; "
        "do not invent or imply the existence of other cases. "
        "Never invent sources or citations."
    )

    section_b = ""
    if has_cases and B_block.strip():
        section_b = (
            "\n\nSection B — Case Excerpts (context only; do not change your decision based on these):\n---\n"
            + B_block
        )

    # IMPORTANT: Do not ask the model to generate a Sources section.
    # We will append deterministic sources ourselves after completion.
    user = (
        "User Situation:\n---\n" + user_situation.strip() + "\n\n"
        "Section A — Legislation Excerpts (authoritative; use these to assess and justify):\n---\n"
        + (A_block.strip() if A_block.strip() else "(none retrieved)")
        + section_b
        + "\n\nYour tasks:\n"
        "1) Assessment under the Law & Policy (use ONLY Section A):\n"
        "   - Return one of: 'very likely', 'likely', 'borderline', or 'unlikely'.\n"
        "   - Provide 3–5 bullets mapping the user's facts to legal elements. Quote sparingly.\n"
        "   - Cite ONLY using the bracket references provided in the excerpts (e.g., [A 1], [B 1]).\n"
        "2) Confidence: repeat the likelihood from #1.\n"
    )

    if has_cases:
        user += (
            "3) Similar Tribunal Decisions (context only): list 1–3 items drawn ONLY from the case excerpts above; "
            "1–2 lines each; include outcome; cite ONLY using [B n].\n"
        )
    else:
        user += "3) Do not include a 'Similar Tribunal Decisions' section (no case excerpts were provided).\n"

    user += (
        "4) Next Steps: 3–5 actionable steps consistent with your assessment.\n"
        "5) Do NOT include a Sources section.\n"
        "6) Disclaimer: one line that this is general information, not legal advice."
    )

    return {"system": system, "user": user}


# ----------------------------
# Chroma setup + retrieval
# ----------------------------
def _get_chroma_client() -> chromadb.PersistentClient:
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

def _get_embedding_fn() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )

def _get_collection(client: chromadb.PersistentClient, name: str):
    """Get an existing collection and attach embedding_function for consistent query embedding."""
    try:
        return client.get_collection(
            name=name,
            embedding_function=_get_embedding_fn(),
        )
    except Exception as e:
        raise RuntimeError(
            f"Chroma collection '{name}' not found in '{PERSIST_DIR}'. "
            f"Expected collections: '{COL_CASES}', '{COL_LEGIS}'."
        ) from e

def _format_hits(
    docs: List[str],
    metas: List[Dict[str, Any]] | None,
    prefix: str,
) -> tuple[str, dict[int, str]]:
    """
    Format retrieved chunks as a readable block with deduped source IDs.
    Returns:
      - formatted block text
      - id_to_source mapping: {1: "filename-or-url", ...}
    """
    if not docs:
        return "", {}

    metas = metas or [{} for _ in docs]

    source_to_id: dict[str, int] = {}
    id_to_source: dict[int, str] = {}
    out: list[str] = []
    next_id = 1

    for doc, meta in zip(docs, metas):
        raw_src = meta.get("source_url") or meta.get("source_path") or "unknown_source"

        # If it's a URL, keep as-is. If it's a file path, keep only the basename.
        if isinstance(raw_src, str) and (raw_src.startswith("http://") or raw_src.startswith("https://")):
            src = raw_src
        else:
            src = os.path.basename(str(raw_src))

        if src not in source_to_id:
            source_to_id[src] = next_id
            id_to_source[next_id] = src
            next_id += 1

        ref_id = source_to_id[src]

        out.append(f"[{prefix} {ref_id}] Source: {src}")
        out.append(doc.strip())
        out.append("")

    return "\n".join(out).strip(), id_to_source

def retrieve_context(user_query: str) -> tuple[str, str, dict[int, str], dict[int, str]]:
    """
    Query both collections using query_texts.
    Returns:
      (A_block_legislation, B_block_cases, A_sources, B_sources)
    """
    client = _get_chroma_client()

    legis_col = _get_collection(client, COL_LEGIS)
    cases_col = _get_collection(client, COL_CASES)

    legis_res = legis_col.query(query_texts=[user_query], n_results=TOP_K_LEGIS)
    cases_res = cases_col.query(query_texts=[user_query], n_results=TOP_K_CASES)

    legis_docs = (legis_res.get("documents") or [[]])[0]
    legis_metas = (legis_res.get("metadatas") or [[]])[0]
    cases_docs = (cases_res.get("documents") or [[]])[0]
    cases_metas = (cases_res.get("metadatas") or [[]])[0]

    A_block, A_sources = _format_hits(legis_docs, legis_metas, "A")
    B_block, B_sources = _format_hits(cases_docs, cases_metas, "B")

    return A_block, B_block, A_sources, B_sources

def _render_sources(A_sources: dict[int, str], B_sources: dict[int, str]) -> str:
    lines = ["5) Sources:"]

    for i, src in sorted(A_sources.items()):
        lines.append(f"- [A {i}] Source: {src}")

    for i, src in sorted(B_sources.items()):
        lines.append(f"- [B {i}] Source: {src}")

    return "\n".join(lines)

def _strip_model_sources_section(text: str) -> str:
    """
    Defensive cleanup: if the model outputs its own '5) Sources:' section anyway,
    remove it so we can append our deterministic one.
    We remove everything from a line starting with '5) Sources:' up to (but not including)
    a line starting with '6) Disclaimer:' if present; otherwise to end of text.
    """
    pattern = r"\n5\)\s*Sources:\s*\n.*?(?=\n6\)\s*Disclaimer:|\Z)"
    return re.sub(pattern, "", text, flags=re.DOTALL)

# ----------------------------
# LLM call
# ----------------------------
def run_rag_analysis(user_situation: str) -> str:
    """
    End-to-end:
      - retrieve top chunks
      - build prompt
      - call LLM
      - append deterministic sources
    """
    A_block, B_block, A_sources, B_sources = retrieve_context(user_situation)
    has_cases = bool(B_block.strip())

    prompt = _build_prompt(
        user_situation=user_situation,
        A_block=A_block,
        B_block=B_block,
        has_cases=has_cases,
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()

    # Remove any model-generated Sources section (defensive)
    answer = _strip_model_sources_section(answer).strip()

    # Append deterministic Sources section
    sources_block = _render_sources(A_sources, B_sources)

    # Ensure Sources appear before Disclaimer if Disclaimer exists, otherwise append at end
    if re.search(r"(?m)^6\)\s*Disclaimer:", answer):
        answer = re.sub(
            r"(?m)^6\)\s*Disclaimer:",
            sources_block + "\n\n6) Disclaimer:",
            answer,
            count=1,
        )
        return answer.strip()

    return (answer + "\n\n" + sources_block).strip()


# s = "I was called a faggot by my coworker"
# print(run_rag_analysis(s))
