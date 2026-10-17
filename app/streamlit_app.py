import glob, json, os
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="IssueScout", layout="wide")
st.title("IssueScout")
st.caption("ML predicts, rules and search decide, the LLM only explains.")

with st.sidebar:
    st.header("Your profile")
    skills = st.text_input("Skills (comma separated)", "go, docker, python")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"])
    hpw = st.number_input("Hours per week", 1, 40, 6)
    weeks = st.number_input("Weeks available", 1, 26, 3)
    k = st.slider("Issues in plan", 1, 6, 3)
    avoid = st.text_input("Avoid keywords", "scheduler")
    use_llm = st.checkbox("LLM explanations (needs API key)", False)
    go = st.button("Find my issues", type="primary")

if go:
    body = {"skills": [s.strip().lower() for s in skills.split(",") if s.strip()], "level": level,
            "hours_per_week": hpw, "weeks": weeks, "k": k,
            "avoid": [a.strip() for a in avoid.split(",") if a.strip()], "use_llm": use_llm}
    with st.spinner("Scoring issues..."):
        try:
            r = requests.post(f"{API}/recommend", json=body, timeout=300)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except Exception as e:
            st.error(f"API error: {e}. Is `uvicorn api.main:app` running?")

tab_plan, tab_map, tab_report = st.tabs(["Plan", "Repo map", "Model report"])

with tab_plan:
    res = st.session_state.get("result")
    if not res:
        st.info("Fill the sidebar and click **Find my issues**.")
    elif not res["plan"]:
        st.warning(res.get("note", "No plan found."))
        st.json(res.get("rejected", {}))
    else:
        st.success(f"{res['planned_hours']}h planned out of {res['budget_hours']}h available")
        for i in res["plan"]:
            st.subheader(f"#{i['rank']}  {i['repo']} #{i['number']}: {i['title']}")
            a, b, c, d = st.columns(4)
            a.metric("Difficulty", f"{i['difficulty']} ({i['diff_conf']:.0%})")
            b.metric("P(merged)", f"{i['p_merge']:.0%}")
            c.metric("Est. effort", f"{i['est_hours']} h")
            d.metric("Maintainer reply", f"{i['reply_days']} d" if i["reply_days"] is not None else "n/a")
            st.link_button("Open on GitHub", i["url"])
            with st.expander("Why this issue? (rules, CSP check, explanation)"):
                st.markdown("**Rules fired**")
                for rule in i["rules"] or ["none"]:
                    st.write("- " + rule)
                st.markdown("**CSP check**  " + "  ".join(f"{'✅' if v else '❌'} {k}" for k, v in i["csp_check"].items()))
                st.markdown("**Skills needed:** " + (", ".join(i["skills"]) or "none"))
                st.write(i["explanation"])
            f1, f2, f3 = st.columns(3)
            for col, label, verdict in [(f1, "I tried this", "tried"), (f2, "Too hard", "too_hard"), (f3, "Good pick", "good")]:
                if col.button(label, key=f"{verdict}_{i['id']}"):
                    requests.post(f"{API}/feedback", json={"issue_id": i["id"], "verdict": verdict})
                    st.toast("Feedback saved")
        with st.expander("Search statistics"):
            st.write("Issues rejected by constraints:", res["rejected"])
            st.write(f"CSP nodes explored: forward checking + MRV = {res['csp_nodes_fc_mrv']}, plain backtracking = {res['csp_nodes_plain']}")

with tab_map:
    p = "data/processed/repo_flags.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["cluster"] = df["cluster"].astype(str)
        st.scatter_chart(df, x="pc1", y="pc2", color="cluster")
        st.dataframe(df[df.abandoned][["full_name", "days_since_push"]], use_container_width=True)
    else:
        st.info("Run `python cli.py cluster` first.")

with tab_report:
    if os.path.exists("report/metrics.json"):
        for name, rows in json.load(open("report/metrics.json")).items():
            st.subheader(name)
            st.dataframe(pd.DataFrame(rows).T)
    for img in sorted(glob.glob("report/*.png")):
        st.image(img, caption=os.path.basename(img))