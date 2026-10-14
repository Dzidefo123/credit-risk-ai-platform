"""Analyst interface (Streamlit). Run: uv run --group ui streamlit run src/crisk/ui/app.py

Configuration: CRISK_API_URL (default http://127.0.0.1:8000). The API key is entered in the sidebar
(or read from CRISK_API_KEY) and used for every request, so the API enforces permissions and records
who did what. The interface is decision support only: it shows a risk decile and the model's reasons,
records human review actions, and never makes an automated credit decision.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from crisk.ui.client import ApiClient, ApiError

st.set_page_config(page_title="Credit risk analyst", layout="wide")

# ---- connection -----------------------------------------------------------------------------------------------
st.sidebar.header("Connection")
api_url = st.sidebar.text_input("API URL", os.environ.get("CRISK_API_URL", "http://127.0.0.1:8000"))
api_key = st.sidebar.text_input("API key", os.environ.get("CRISK_API_KEY", ""), type="password")
page = st.sidebar.radio("Page", ["Score application", "Review queue", "Monitoring", "Documents", "Copilot",
                               "Model & governance"])

if not api_key:
    st.info("Enter your API key in the sidebar. Keys are issued with `crisk create-api-key --role analyst`.")
    st.stop()
client = ApiClient(api_url, api_key)

try:
    model = client.model()
except ApiError as e:
    st.error(f"Cannot reach the scoring API: {e}")
    st.stop()
except OSError as e:
    st.error(f"Cannot reach the scoring API at {api_url}: {e}")
    st.stop()

# ---- governance banner (always visible) -------------------------------------------------------------------------
origin = model.get("data_origin")
use = model.get("approved_use")
st.title("Credit risk analyst workspace")
banner = (f"**{origin}** · model `{model.get('model_id')}` · approved use: **{use}**  \n"
          f"{model.get('approved_use_text', '')}")
if origin == "SYNTHETIC":
    st.warning("SYNTHETIC data: results test the system only.  \n" + banner)
elif use != "pd_level":
    st.warning(banner + "  \nRead the **risk decile**; the PD figure is shown for transparency only.")
else:
    st.info(banner)
st.caption(f"Signed in as {model['caller']['name']} ({model['caller']['role']}). Decision support only: "
           "no automated approve/decline.")

DEFAULT_APP = dict(credit_score=720, orig_dti=36, orig_ltv=80, orig_cltv=80, orig_interest_rate=6.5,
                   orig_loan_term=360, mi_pct=0, num_units=1, orig_upb=300000.0, num_borrowers=2,
                   occupancy_status="P", channel="R", property_type="SF", loan_purpose="P",
                   first_time_homebuyer_flag="N", property_state="TX")
CODES = {"occupancy_status": ["P", "I", "S"], "channel": ["R", "B", "C", "T"],
         "property_type": ["SF", "PU", "CO", "MH", "CP"], "loan_purpose": ["P", "C", "N", "R"],
         "first_time_homebuyer_flag": ["Y", "N"]}


def show_result(r: dict) -> None:
    c1, c2, c3 = st.columns(3)
    if r["status"] != "scored":
        st.error(f"No score produced ({r['status']}): {', '.join(r['dq_flags'])}")
        return
    c1.metric("Risk decile (1 = lowest, 10 = highest)", r.get("risk_decile") or "n/a")
    c2.metric("Model PD (12m, 90+ DPD)", f"{100 * r['pd']:.2f}%", help="Not validated for PD-level use")
    c3.metric("Score ID", r["score_id"][:12] + "…")
    if r["reason_codes"]:
        st.subheader("Main factors raising the estimate (model attributions, not adverse-action reasons)")
        st.table(pd.DataFrame(r["reason_codes"])[["code", "text", "contribution_logit"]])
    if r["dq_flags"]:
        st.caption("Data-quality flags: " + ", ".join(r["dq_flags"]))


# ---- pages ---------------------------------------------------------------------------------------------------------
if page == "Score application":
    st.header("Score an application")
    with st.form("app"):
        cols = st.columns(4)
        app = {}
        numeric = ["credit_score", "orig_dti", "orig_ltv", "orig_cltv", "orig_interest_rate", "orig_loan_term",
                   "mi_pct", "num_units", "orig_upb", "num_borrowers"]
        for i, f in enumerate(numeric):
            v = cols[i % 4].text_input(f, str(DEFAULT_APP[f]), help="Leave blank if not available")
            app[f] = None if v.strip() == "" else float(v) if f in ("orig_interest_rate", "orig_upb") else int(float(v))
        for i, (f, opts) in enumerate(CODES.items()):
            choice = cols[i % 4].selectbox(f, ["(not available)", *opts], index=opts.index(DEFAULT_APP[f]) + 1)
            app[f] = None if choice == "(not available)" else choice
        if "property_state" in (model.get("input_fields") or []):
            app["property_state"] = cols[1].text_input("property_state", DEFAULT_APP["property_state"]).upper() or None
        if "origination_v2" in str(model.get("feature_set_id")):
            app["first_payment_date"] = str(cols[2].date_input("first_payment_date"))
        submitted = st.form_submit_button("Score")
    if submitted:
        allowed = set(model.get("input_fields") or app)  # send only the fields the served model accepts
        app = {k: v for k, v in app.items() if k in allowed}
        try:
            show_result(client.score(app, idempotency_key=f"ui-{uuid.uuid4().hex}"))
        except ApiError as e:
            st.error(str(e))

elif page == "Review queue":
    st.header("Review queue")
    try:
        items = client.scores(limit=200)
    except ApiError as e:
        st.error(str(e))
        st.stop()
    if not items:
        st.info("No scores yet.")
        st.stop()
    df = pd.DataFrame(items)
    status = st.multiselect("Review status", sorted(df.review_status.unique()), default=["pending"]
                            if "pending" in set(df.review_status) else None)
    view = df[df.review_status.isin(status)] if status else df
    st.dataframe(view[["score_id", "scored_at", "status", "risk_decile", "requested_by", "review_status",
                       "override_decision"]], use_container_width=True, hide_index=True)
    sid = st.selectbox("Open score", view.score_id.tolist())
    if sid:
        d = client.score_detail(sid)
        show_result(d["response"])
        with st.expander("Application inputs (as scored)"):
            st.json(d["input_snapshot"])
        if d["reviews"]:
            st.subheader("Review history")
            st.table(pd.DataFrame(d["reviews"])[["reviewed_at", "reviewer", "reviewer_role", "action",
                                                 "override_decision", "rationale"]])
        with st.form("review"):
            action = st.radio("Review action", ["agree", "override", "escalate"], horizontal=True)
            decision = st.selectbox("Override decision (only for override)", ["", "approve", "decline", "refer"])
            rationale = st.text_area("Rationale (required, at least 10 characters)")
            if st.form_submit_button("Record review"):
                try:
                    client.review(sid, action, rationale, decision or None)
                    st.success("Review recorded (append-only; visible to auditors).")
                except ApiError as e:
                    st.error(str(e))
        if st.button("Replay this score (auditor/validator only)"):
            try:
                st.json(client.replay(sid))
            except ApiError as e:
                st.error(str(e))

elif page == "Monitoring":
    st.header("Portfolio monitoring")
    try:
        mon = client.monitoring()
    except ApiError as e:
        st.error(str(e))
        st.stop()
    st.caption(f"Stream: {mon['stream']} · rank reference: {mon['reference'].get('rank_reference')} · "
               f"generated {mon['generated_at']}")
    al = pd.DataFrame(mon["alerts"])
    st.subheader("Alerts (de-duplicated)")
    if al.empty:
        st.success("No alerts.")
    else:
        st.dataframe(al[["severity", "alert_id", "first_period", "last_period", "periods", "peak_value", "owner",
                         "action", "message"]], use_container_width=True, hide_index=True)
    months = pd.DataFrame(mon["months"]).set_index("month")
    th = mon["thresholds"]
    st.subheader(f"Score drift (PSI vs rank reference; watch ≥ {th['psi_watch']}, alert ≥ {th['psi_alert']})")
    st.line_chart(months[["score_psi"]])
    st.subheader("Share of applications above the reference top-decile cut-off (expected 10%)")
    st.line_chart(months[["share_top_reference_decile"]])
    st.subheader("Monthly volume")
    st.bar_chart(months[["volume"]])
    q = pd.DataFrame(mon["quarters"])
    if not q.empty:
        st.subheader("Matured outcomes by quarter")
        st.dataframe(q, use_container_width=True, hide_index=True)
    fp = pd.DataFrame({m: r["feature_psi"] for m, r in zip(months.index, mon["months"], strict=True)}).T
    st.subheader("Input drift (feature PSI by month)")
    st.dataframe(fp.round(3), use_container_width=True)

elif page == "Documents":
    st.header("Document review (SYNTHETIC documents only)")
    st.caption("Documents are untrusted content: instruction-like text is flagged and ignored. Extracted values never "
               "become model inputs; only a validated packet yields verified inputs, for comparison with the "
               "application.")
    with st.expander("Submit a packet"):
        fixture_dir = Path(os.environ.get("CRISK_DOC_FIXTURES", "tests/fixtures/underwriting_docs_v1")) / "packets"
        samples = sorted(fixture_dir.glob("*.json")) if fixture_dir.exists() else []
        choice = st.selectbox("Fictional sample packet", ["(paste JSON instead)"] + [x.stem for x in samples])
        raw = "" if choice != "(paste JSON instead)" else st.text_area(
            "Packet JSON", '{"data_origin": "SYNTHETIC", "documents": [{"doc_type": "paystub", "pages": ["..."]}]}')
        if st.button("Submit packet"):
            try:
                packet = (json.loads((fixture_dir / f"{choice}.json").read_text(encoding="utf-8")) if raw == ""
                          else json.loads(raw))
                res = client.submit_documents(packet)
                st.session_state["packet_id"] = res["packet_id"]
                st.success(f"Packet {res['packet_id']} submitted: status {res['status']}")
            except (ApiError, ValueError) as e:
                st.error(str(e))
    try:
        items = client.documents()
    except ApiError as e:
        st.error(str(e))
        st.stop()
    if not items:
        st.info("No packets yet.")
        st.stop()
    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
    ids = [x["packet_id"] for x in items]
    default = ids.index(st.session_state["packet_id"]) if st.session_state.get("packet_id") in ids else 0
    pid = st.selectbox("Packet", ids, index=default,
                       format_func=lambda i: next(f"{x['packet_ref'] or i} · {x['status']}" for x in items
                                                  if x["packet_id"] == i))
    d = client.document(pid, include_documents=True)
    {"validated": st.success, "needs_review": st.warning, "quarantined": st.error,
     "rejected": st.error}[d["status"]](f"Status: {d['status']}")
    if d["flags"]:
        st.subheader("Flags")
        st.dataframe(pd.DataFrame(d["flags"]), use_container_width=True, hide_index=True)
    st.subheader("Extracted fields (page and quote references)")
    rows = [{"field": f"{doc}.{f}", **v} for doc, m in d["fields"].items() for f, v in m.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for doc in d.get("documents", []):
        with st.expander(f"Source: {doc['doc_type']} ({len(doc['pages'])} pages)"):
            for i, pg in enumerate(doc["pages"], start=1):
                st.markdown(f"**Page {i}**")
                st.code(pg, language=None)
    st.subheader("Reviewer actions")
    with st.form("correct"):
        field = st.selectbox("Field", [r["field"] for r in rows])
        value = st.text_input("Corrected value (as written in the document)")
        rationale = st.text_area("Rationale (required)", key="corr_rationale")
        if st.form_submit_button("Record correction"):
            try:
                st.json(client.correct_document(pid, field, value, rationale)["open_flags"])
            except ApiError as e:
                st.error(str(e))
    with st.form("review"):
        action = st.radio("Decision", ["accept", "reject"], horizontal=True,
                          help="Accept is allowed only for acceptable flags (mismatches, a suspected instruction "
                               "after checking it was ignored). Missing fields and arithmetic errors must be "
                               "corrected.")
        rationale = st.text_area("Rationale (required)", key="rev_rationale")
        if st.form_submit_button("Record decision"):
            try:
                st.success(f"Status: {client.review_document(pid, action, rationale)['status']}")
            except ApiError as e:
                st.error(str(e))
    if st.button("Verified inputs"):
        try:
            st.json(client.verified_inputs(pid))
        except ApiError as e:
            st.error(str(e))

elif page == "Copilot":
    st.header("Analyst copilot")
    st.caption("Answers come only from approved, versioned project documents your role may read, and from read-only "
               "tools. Every answer cites its sources; the copilot abstains when the evidence is missing.")
    question = st.text_input("Question", placeholder="What is the base scenario 12-month expected loss rate?")
    if st.button("Ask") and question.strip():
        try:
            r = client.ask(question)
        except ApiError as e:
            st.error(str(e))
            st.stop()
        if r["outcome"] == "answer":
            st.success(r["answer"])
            if r.get("fallback"):
                st.warning(f"Fallback: {r['fallback']}")
        else:
            st.warning(f"{r['answer']} ({r['outcome']}: {r['reason']})")
        if r["citations"]:
            st.subheader(f"Citations (as of {r.get('as_of')})")
            st.dataframe(pd.DataFrame(r["citations"])[["source_id", "path", "lines", "version", "as_of", "quote"]],
                         use_container_width=True, hide_index=True)
        with st.expander("Retrieved passages and tools"):
            st.json({"retrieved": r["retrieved"], "tools": r["tools"], "llm": r["llm"], "usage": r.get("usage")})
        st.caption(r["label"])

else:
    st.header("Model & governance")
    st.json({k: model.get(k) for k in ("model_id", "model_family", "feature_set_id", "target", "data_origin",
                                       "evidence_note", "approved_use", "approved_use_text", "champion")})
    st.markdown("Evidence: `docs/results/` (baseline, holdout, recalibration and TTC studies) and "
                "`docs/evaluation_plan_v*.md` (pre-registrations).")
