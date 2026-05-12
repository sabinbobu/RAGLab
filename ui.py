"""RAGLab demo UI.

Run with:
    uv run streamlit run ui.py

Requires the FastAPI backend to be running:
    uv run uvicorn raglab.main:app --reload
"""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAGLab", page_icon="🧪", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "experiment_id" not in st.session_state:
    st.session_state.experiment_id = None
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
                    with st.spinner(
                        f"Running RAG pipeline with {DISPLAY_NAMES.get(model, model)}…"
                    ):
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
        exp_provider = st.selectbox("Provider", list(MODELS.keys()), key="exp_prov")
        exp_retriever = st.selectbox(
            "Retriever", ["chroma", "bm25", "hybrid"], key="exp_ret"
        )
        exp_top_k = st.slider("Top K Chunks", 1, 10, 5, key="exp_topk")

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
            with st.spinner(
                f"Running {n_runs} combinations… please wait, this may take a minute."
            ):
                resp = httpx.post(
                    f"{API_BASE}/experiments/run",
                    json={
                        "models": selected_models,
                        "prompt_versions": selected_prompts,
                        "questions": valid_qs,
                        "top_k": exp_top_k,
                        "provider": exp_provider,
                        "retriever": exp_retriever,
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()

            completed = data["total_runs"]
            st.session_state.experiment_id = data["experiment_id"]
            st.session_state.scorecards = None
            st.success(
                f"Experiment complete — {completed}/{n_runs} runs succeeded. "
                "Switch to the **Results** tab to evaluate with Ragas."
            )
            st.code(f"experiment_id: {st.session_state.experiment_id}", language=None)

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

    if not st.session_state.experiment_id:
        st.info("Run an experiment first to see results here.", icon="ℹ️")
    else:
        st.caption(f"Experiment: `{st.session_state.experiment_id}`")

        eval_btn = st.button("Evaluate with Ragas", type="primary", key="eval_btn")

        if eval_btn:
            try:
                with st.spinner(
                    "Evaluating faithfulness with Ragas… "
                    "this makes additional LLM calls."
                ):
                    resp = httpx.post(
                        f"{API_BASE}/experiments/evaluate",
                        params={"experiment_id": st.session_state.experiment_id},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    st.session_state.scorecards = resp.json()

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
            st.success(
                f"Best performer: **{best_name}** "
                f"with prompt **{best['prompt_version']}** — "
                f"faithfulness **{best['faithfulness']:.3f}**",
                icon="🏆",
            )

            # Scorecard table
            df = pd.DataFrame(
                [
                    {
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
