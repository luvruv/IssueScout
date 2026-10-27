import glob, json, os, sys
from pathlib import Path

# Ensure project root is in sys.path when running via streamlit
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IssueScout", layout="wide")
st.title("IssueScout")
st.caption("ML predicts, rules and search decide, the LLM only explains. Every number is a hint: read the issue yourself.")


def need_models():
    if not os.path.exists("models/difficulty.joblib"):
        st.error("No trained model yet. Run `python cli.py train all` first."); st.stop()


with st.sidebar:
    st.header("Your profile")
    skills = st.text_input("Skills (comma separated)", "go, docker, python")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1)
    hpw = st.number_input("Hours per week", 1, 40, 6); weeks = st.number_input("Weeks available", 1, 26, 3)
    k = st.slider("Issues in plan", 1, 6, 3); avoid = st.text_input("Avoid keywords", "")
    uncertain = st.checkbox("Include issues the model is unsure about", True)
    use_llm = st.checkbox("LLM explanations (needs LLM_PROVIDER in .env)", False)
    go = st.button("Find my issues", type="primary")

if go:
    need_models()
    from core.profile import Profile
    from core.recommend import recommend
    p = Profile(skills=[s.strip().lower() for s in skills.split(",") if s.strip()], level=level, hours_per_week=hpw, weeks=weeks, k=k,
                avoid=[a.strip() for a in avoid.split(",") if a.strip()], use_llm=use_llm, include_uncertain=uncertain)
    with st.spinner("Scoring issues..."):
        st.session_state["result"] = recommend(p)

tabs = st.tabs(["Plan", "First-PR kit", "Predict any issue", "Ask the docs", "My outcomes", "Repo map", "Model report"])

with tabs[0]:
    res = st.session_state.get("result")
    if not res:
        st.info("Fill the sidebar and click **Find my issues**.")
    elif not res["plan"]:
        st.warning(res.get("note", "No plan found.")); st.json(res.get("rejected", {}))
    else:
        st.success(f"{res['planned_hours']}h planned out of {res['budget_hours']}h available")
        for i in res["plan"]:
            st.subheader(f"#{i['rank']}  {i['repo']} #{i['number']}: {i['title']}")
            if i["uncertain"]:
                st.warning("The model is unsure about this one (top probability under 50%). Read the issue before committing.")
            a, b, c, d = st.columns(4)
            a.metric("Predicted difficulty", i["difficulty"], f"conf {i['diff_conf']:.0%}", delta_color="off")
            b.metric("P(merged)", f"{i['p_merge']:.0%}"); c.metric("Est. effort", f"{i['est_hours']} h")
            d.metric("Maintainer reply", f"{i['reply_days']} d" if i["reply_days"] is not None else "n/a")
            st.progress(min(1.0, i["probs"]["Easy"]), text=f"P(Easy)={i['probs']['Easy']:.0%}  P(Medium)={i['probs']['Medium']:.0%}  P(Hard)={i['probs']['Hard']:.0%}")
            st.link_button("Open on GitHub", i["url"])
            with st.expander("Why this issue? (rules, CSP check, similar solved issues)"):
                st.markdown("**Rules fired**\n" + "\n".join("- " + r for r in (i["rules"] or ["none"])))
                st.markdown("**CSP check**  " + "  ".join(f"{'✅' if v else '❌'} {kk}" for kk, v in i["csp_check"].items()))
                st.markdown("**Skills needed:** " + (", ".join(i["skills"]) or "none"))
                st.write(i["explanation"]); st.caption(f"explanation source: {i['explanation_meta']['source']}")
                for s in i.get("similar_solved", []):
                    st.markdown(f"- similar solved: [{s['title']}]({s['url']}) was **{s['difficulty']}** (similarity {s['similarity']})")
            f1, f2, f3 = st.columns(3)
            for col, label, verdict in [(f1, "I tried this", "tried"), (f2, "Too hard", "too_hard"), (f3, "Good pick", "good")]:
                if col.button(label, key=f"{verdict}_{i['id']}"):
                    os.makedirs("data", exist_ok=True)
                    open("data/feedback.jsonl", "a").write(json.dumps({"issue_id": i["id"], "verdict": verdict}) + "\n"); st.toast("Feedback saved")
        with st.expander("Search statistics"):
            st.write("Issues rejected:", res["rejected"])
            st.write(f"CSP nodes: FC+MRV+bound = {res['csp_nodes_fc_mrv']}, plain backtracking = {res['csp_nodes_plain']}")

with tabs[1]:
    ref = st.text_input("Issue (owner/repo#123 or URL)", key="kit_ref"); kllm = st.checkbox("Use LLM for doc answers", False, key="kit_llm")
    if st.button("Build my first-PR kit") and ref:
        from llm.first_pr_kit import build_kit, render_markdown, save
        try:
            with st.spinner("Reading the repo's contributor docs..."):
                kit = build_kit(ref, use_llm=kllm)
            st.markdown(render_markdown(kit)); st.caption(f"saved to {save(kit)}")
        except Exception as e:
            st.error(f"Could not build the kit: {e}")

with tabs[2]:
    pref = st.text_input("Issue to classify (owner/repo#123 or URL)", key="pred_ref")
    if st.button("Predict difficulty") and pref:
        need_models()
        from ml.predict import predict_issue
        try:
            r = predict_issue(pref); st.subheader(f"{r['predicted']}  (confidence {r['confidence']:.0%})")
            st.bar_chart(pd.Series(r["probabilities"]))
            if r["uncertain"]: st.warning("Model is unsure. " + r["note"])
            st.write("Most similar solved issues:"); st.dataframe(pd.DataFrame(r["similar_solved"]))
        except Exception as e:
            st.error(str(e))

with tabs[3]:
    rrepo = st.text_input("Repo (owner/name)", key="ask_repo"); q = st.text_input("Question", "How do I run the tests?", key="ask_q")
    if st.button("Ask") and rrepo:
        from llm.rag import ask
        try:
            a = ask(rrepo, q); st.write(a["answer"]); st.caption(f"{a['mode']} | {a['backend']} | groundedness {a['grounded']}")
            if a.get("warning"): st.warning(a["warning"])
            for s in a["sources"]: st.markdown(f"- `{s['path']}` > {s['heading']}")
        except Exception as e:
            st.error(str(e))

with tabs[4]:
    from core import outcomes
    st.json(outcomes.summary())
    oid = st.text_input("Issue id (owner/repo#123)", key="o_id"); stt = st.selectbox("Status", outcomes.STATUSES); hrs = st.number_input("Hours spent", 0.0, 500.0, 0.0)
    if st.button("Log outcome") and oid:
        outcomes.add(oid, stt, hrs or None); st.success("logged")
    st.dataframe(pd.DataFrame(outcomes.load()))

with tabs[5]:
    p = "data/processed/repo_flags.csv"
    if os.path.exists(p):
        df = pd.read_csv(p); df["cluster"] = df["cluster"].astype(str)
        st.scatter_chart(df, x="pc1", y="pc2", color="cluster"); st.dataframe(df[df.abandoned][["full_name", "days_since_push"]])
    else:
        st.info("Run `python cli.py cluster` first.")

with tabs[6]:
    if os.path.exists("report/RESULTS.md"):
        st.markdown(open("report/RESULTS.md", encoding="utf-8").read())
    for img in sorted(glob.glob("report/*.png")):
        st.image(img, caption=os.path.basename(img))