"""RAGLab demo UI.

Run with:
    uv run streamlit run ui.py

Requires the FastAPI backend to be running:
    uv run uvicorn raglab.main:app --reload
"""

from __future__ import annotations

import itertools
from collections import Counter

import chromadb
import httpx
import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAGLab", page_icon="🧪", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "experiment_ids" not in st.session_state:
    st.session_state.experiment_ids = []  # one id per provider group
if "scorecards" not in st.session_state:
    st.session_state.scorecards = None

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"],
}
ALL_MODELS = [m for ms in MODELS.values() for m in ms]

DISPLAY_NAMES: dict[str, str] = {
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o": "GPT-4o",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
}

# Reverse-lookup: model id → provider name
MODEL_TO_PROVIDER: dict[str, str] = {
    m: provider for provider, models in MODELS.items() for m in models
}


# ── API health check ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _api_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE}/", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


if not _api_reachable():
    st.warning(
        f"API not reachable at **{API_BASE}** — "
        "start it with `uv run uvicorn raglab.main:app --reload`",
        icon="⚠️",
    )

# ── Header ────────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([8, 2])
with hdr_left:
    st.title("RAGLab")
    st.caption("Compare LLMs · retrievers · prompts · faithfulness · cost · latency")
with hdr_right:
    st.markdown(
        "<div style='display:flex;gap:8px;justify-content:flex-end;padding-top:1.2rem'>"
        "<span style='background:#e0e7ff;color:#3730a3;padding:3px 12px;"
        "border-radius:9999px;font-size:0.75rem;font-weight:600'>v1.0</span>"
        "<span style='background:#f0fdf4;color:#15803d;padding:3px 12px;"
        "border-radius:9999px;font-size:0.75rem;font-weight:600'>FastAPI</span>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ── Corpus ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _corpus_stats() -> list[dict]:
    """Return per-document stats from ChromaDB (cached for the session)."""
    try:
        client = chromadb.PersistentClient(path=".chroma")
        col = client.get_collection(name="raglab")
        result = col.get(include=["metadatas"])
        metas = result["metadatas"] or []
        chunk_counts = Counter(str(m.get("source", "unknown")) for m in metas)
        page_counts = {}
        for m in metas:
            src = str(m.get("source", "unknown"))
            page_counts[src] = max(page_counts.get(src, 0), int(m.get("page", 0)))
        return [
            {"Document": src, "Pages": page_counts[src], "Chunks": cnt}
            for src, cnt in sorted(chunk_counts.items())
        ]
    except Exception:
        return []


docs = _corpus_stats()
if docs:
    st.markdown(f"**Corpus** — {len(docs)} document(s) ingested")
    st.dataframe(
        pd.DataFrame(docs),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No documents ingested yet. "
        "Run `uv run raglab ingest data/` to populate the corpus.",
        icon="📂",
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
query_tab, experiment_tab, results_tab = st.tabs(
    ["🔍  Query", "⚗️  Experiment", "📊  Results"]
)


# =============================================================================
# QUERY TAB
# =============================================================================
with query_tab:
    col_cfg, col_ans = st.columns([1, 1.6], gap="large")

    with col_cfg:
        st.markdown("##### Configuration")

        provider = st.selectbox("Provider", list(MODELS.keys()), key="q_provider")
        model = st.selectbox("Model", MODELS[provider], key="q_model")

        retriever_strategy = st.radio(
            "Retriever Strategy",
            ["chroma", "bm25", "hybrid"],
            horizontal=True,
            key="q_retriever",
        )
        prompt_version = st.radio(
            "Prompt Version",
            ["v1", "v2"],
            horizontal=True,
            captions=["Precise / Citations", "Conversational"],
            key="q_prompt",
        )
        top_k = st.slider(
            "Top K Chunks", min_value=1, max_value=10, value=5, key="q_topk"
        )

        st.markdown("##### Question")
        question = st.text_area(
            "question",
            label_visibility="collapsed",
            placeholder="Ask a question about your documents…",
            height=130,
            key="q_question",
        )
        ask_btn = st.button(
            "Ask", type="primary", use_container_width=True, key="q_ask"
        )

    with col_ans:
        st.markdown("##### Answer")

        if ask_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    model_name = DISPLAY_NAMES.get(model, model)
                    with st.status(
                        "Running RAG pipeline…", expanded=True
                    ) as rag_status:
                        st.write(
                            f"**Step 1 / 2** — Retrieving {top_k} chunks"
                            f" via **{retriever_strategy}**"
                        )
                        resp = httpx.post(
                            f"{API_BASE}/query",
                            json={
                                "question": question,
                                "provider": provider,
                                "model": model,
                                "prompt_version": prompt_version,
                                "top_k": top_k,
                                "retriever": retriever_strategy,
                            },
                            timeout=60,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        n_src = len(data.get("sources", []))
                        st.write(f"**Step 1 / 2** — Retrieved {n_src} chunks ✓")
                        st.write(
                            f"**Step 2 / 2** — Generated answer with"
                            f" **{model_name}** —"
                            f" {data['latency_ms']:.0f} ms"
                            f" · ${data['cost_usd']:.5f} ✓"
                        )
                        rag_status.update(
                            label="Pipeline complete",
                            state="complete",
                            expanded=False,
                        )

                    # Metrics row
                    m1, m2 = st.columns(2)
                    m1.metric("Latency", f"{data['latency_ms']:.0f} ms")
                    m2.metric("Cost", f"${data['cost_usd']:.5f}")

                    st.markdown(data["answer"])

                    sources = data.get("sources", [])
                    if sources:
                        st.markdown(f"**Sources** ({len(sources)})")
                        for chunk in sources:
                            label = (
                                f"**{chunk['source']}** — "
                                f"page {chunk['page']} · score `{chunk['score']:.3f}`"
                            )
                            with st.expander(label):
                                st.caption(chunk["text"])

                except httpx.HTTPStatusError as exc:
                    detail = exc.response.json().get("detail", str(exc))
                    st.error(f"API error: {detail}")
                except Exception as exc:
                    st.error(f"Error: {exc}")
        else:
            st.markdown(
                "<div style='color:#9ca3af;text-align:center;"
                "margin-top:5rem;font-size:0.9rem'>"
                "Configure settings on the left and click "
                "<strong>Ask</strong> to see the answer here."
                "</div>",
                unsafe_allow_html=True,
            )


# =============================================================================
# EXPERIMENT TAB
# =============================================================================
with experiment_tab:
    st.markdown("##### Experiment Matrix")
    st.caption(
        "Each combination of model × prompt version × question becomes one run. "
        "Results are persisted to SQLite and evaluated with Ragas in the Results tab."
    )

    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        selected_models = st.multiselect(
            "Models",
            options=ALL_MODELS,
            default=["gpt-4o-mini"],
            format_func=lambda m: DISPLAY_NAMES.get(m, m),
            key="exp_models",
        )
        selected_prompts = st.multiselect(
            "Prompt Versions",
            options=["v1", "v2"],
            default=["v1", "v2"],
            key="exp_prompts",
        )

    with col_e2:
        exp_retriever = st.selectbox(
            "Retriever", ["chroma", "bm25", "hybrid"], key="exp_ret"
        )
        exp_top_k = st.slider("Top K Chunks", 1, 10, 5, key="exp_topk")
        st.caption(
            "Provider is auto-detected from the selected models — "
            "mix OpenAI and Anthropic freely."
        )

    st.markdown("##### Questions")
    n_q = int(
        st.number_input(
            "Number of questions", min_value=1, max_value=5, value=2, key="exp_nq"
        )
    )
    questions: list[str] = []
    for i in range(n_q):
        q = st.text_area(
            f"Q{i + 1}",
            key=f"exp_q_{i}",
            placeholder=f"Question {i + 1}…",
            height=75,
            label_visibility="collapsed",
        )
        questions.append(q)

    valid_qs = [q for q in questions if q.strip()]
    n_runs = len(selected_models) * len(selected_prompts) * len(valid_qs)

    if n_runs > 0:
        st.info(
            f"**{n_runs} runs** — "
            f"{len(selected_models)} model(s) × "
            f"{len(selected_prompts)} prompt(s) × "
            f"{len(valid_qs)} question(s)",
            icon="ℹ️",
        )
    if n_runs > 8:
        st.warning(
            "Large matrices can take several minutes — API latency adds up.", icon="⚠️"
        )

    run_btn = st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
        disabled=(n_runs == 0),
        key="exp_run",
    )

    if run_btn:
        try:
            # Group selected models by provider so each gets one API call
            provider_groups: dict[str, list[str]] = {}
            for m in selected_models:
                prov = MODEL_TO_PROVIDER[m]
                provider_groups.setdefault(prov, []).append(m)

            new_ids: list[str] = []
            total_completed = 0

            with st.status(
                f"Running {n_runs} combinations…", expanded=True
            ) as exp_status:
                for prov, prov_models in provider_groups.items():
                    n_prov = len(prov_models) * len(selected_prompts) * len(valid_qs)
                    st.write(f"**{prov.title()} — {n_prov} run(s):**")
                    for m, p, q in itertools.product(
                        prov_models, selected_prompts, valid_qs
                    ):
                        st.write(
                            f"· {DISPLAY_NAMES.get(m, m)} × **{p}**" f" × `{q[:55]}`"
                        )
                st.caption("Calling the API — one LLM request per run. Please wait.")
                for prov, prov_models in provider_groups.items():
                    resp = httpx.post(
                        f"{API_BASE}/experiments/run",
                        json={
                            "models": prov_models,
                            "prompt_versions": selected_prompts,
                            "questions": valid_qs,
                            "top_k": exp_top_k,
                            "provider": prov,
                            "retriever": exp_retriever,
                        },
                        timeout=300,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    new_ids.append(data["experiment_id"])
                    total_completed += data["total_runs"]

                    # Render per-run log mirroring the server-side output
                    llm_op = (
                        "openai.chat.completions"
                        if prov == "openai"
                        else "anthropic.client.messages"
                    )
                    log_lines = []
                    for run in data.get("runs", []):
                        q = run["question"][:55]
                        log_lines.append(
                            f"  → {run['model']} | " f"{run['prompt_version']} | {q}"
                        )
                        log_lines.append("      chroma.query              ✓")
                        log_lines.append(
                            f"      {llm_op:<28} ✓"
                            f"  {run['latency_ms']:.0f} ms"
                            f"  ·  ${run['cost_usd']:.5f}"
                        )
                    if log_lines:
                        st.code("\n".join(log_lines), language=None)
                exp_status.update(
                    label=f"{total_completed}/{n_runs} runs completed ✓",
                    state="complete",
                    expanded=False,
                )

            st.session_state.experiment_ids = new_ids
            st.session_state.scorecards = None
            st.success("Switch to the **Results** tab to evaluate with Ragas.")
            for eid in new_ids:
                st.code(f"experiment_id: {eid}", language=None)

        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", str(exc))
            st.error(f"API error: {detail}")
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")


# =============================================================================
# RESULTS TAB
# =============================================================================
with results_tab:
    st.markdown("##### Scorecards")

    if not st.session_state.experiment_ids:
        st.info("Run an experiment first to see results here.", icon="ℹ️")
    else:
        for eid in st.session_state.experiment_ids:
            st.caption(f"Experiment: `{eid}`")

        eval_btn = st.button("Evaluate with Ragas", type="primary", key="eval_btn")

        if eval_btn:
            try:
                all_scorecards: list[dict] = []
                with st.status("Evaluating with Ragas…", expanded=True) as eval_status:
                    st.write("**Step 1 / 3** — Loading runs from SQLite")
                    st.write(
                        "**Step 2 / 3** — Scoring answers with"
                        " **Faithfulness** (LLM-as-judge via OpenAI)"
                    )
                    st.write("**Step 3 / 3** — Building scorecards")
                    for eid in st.session_state.experiment_ids:
                        resp = httpx.post(
                            f"{API_BASE}/experiments/evaluate",
                            params={"experiment_id": eid},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        all_scorecards.extend(resp.json())
                    n = len(all_scorecards)
                    eval_status.update(
                        label=f"Evaluation complete — {n} scorecard(s) ✓",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.scorecards = all_scorecards

            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", str(exc))
                st.error(f"API error: {detail}")
                st.session_state.scorecards = None
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.session_state.scorecards = None

        if st.session_state.scorecards:
            scorecards = sorted(
                st.session_state.scorecards,
                key=lambda s: s["faithfulness"],
                reverse=True,
            )

            best = scorecards[0]
            best_name = DISPLAY_NAMES.get(best["model"], best["model"])
            best_prov = MODEL_TO_PROVIDER.get(best["model"], "").title()
            st.success(
                f"Best performer: **{best_name}** ({best_prov})"
                f" with prompt **{best['prompt_version']}** — "
                f"faithfulness **{best['faithfulness']:.3f}**",
                icon="🏆",
            )

            # Scorecard table — Provider column makes cross-provider comparison clear
            df = pd.DataFrame(
                [
                    {
                        "Provider": MODEL_TO_PROVIDER.get(s["model"], "—").title(),
                        "Model": DISPLAY_NAMES.get(s["model"], s["model"]),
                        "Prompt": s["prompt_version"],
                        "Faithfulness": round(s["faithfulness"], 3),
                        "Avg Cost ($)": f"${s['avg_cost_usd']:.5f}",
                        "Avg Latency (ms)": f"{s['avg_latency_ms']:.0f}",
                        "Runs": s["run_count"],
                    }
                    for s in scorecards
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Charts
            chart_col1, chart_col2 = st.columns(2, gap="large")

            with chart_col1:
                st.markdown("**Faithfulness by Configuration**")
                faith_df = pd.DataFrame(
                    {
                        "Config": [
                            f"{DISPLAY_NAMES.get(s['model'], s['model'])[:15]}"
                            f" / {s['prompt_version']}"
                            for s in scorecards
                        ],
                        "Faithfulness": [s["faithfulness"] for s in scorecards],
                    }
                ).set_index("Config")
                st.bar_chart(faith_df)

            with chart_col2:
                st.markdown("**Average Latency (ms)**")
                lat_df = pd.DataFrame(
                    {
                        "Config": [
                            f"{DISPLAY_NAMES.get(s['model'], s['model'])[:15]}"
                            f" / {s['prompt_version']}"
                            for s in scorecards
                        ],
                        "Avg Latency (ms)": [s["avg_latency_ms"] for s in scorecards],
                    }
                ).set_index("Config")
                st.bar_chart(lat_df)
