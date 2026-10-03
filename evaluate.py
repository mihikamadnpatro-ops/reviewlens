"""
Step 4: evaluate ReviewLens on 20 test questions.

Run:
    pip install openpyxl
    python evaluate.py

Output: evaluation_results.xlsx (a 'summary' sheet and a 'results' sheet).

Set RUN_LLM = False to test only routing + retrieval (no Gemini calls, uses no quota).
"""
import re
import time

import pandas as pd

import graph

RUN_LLM = True        # False = skip answer generation (retrieval-only test)
PAUSE_SECONDS = 4     # pause between questions to stay inside free-tier rate limits

# question, expected route, topic keywords (should appear in retrieved reviews),
# theme keywords (at least one should appear in the answer)
TESTS = [
    ("What do customers complain about regarding coffee?", "complaint", ["coffee"], ["bitter", "weak", "taste", "stale", "flavor"]),
    ("What do people like about the tea?", "praise", ["tea"], ["taste", "flavor", "aroma", "smell"]),
    ("What are the problems with packaging?", "complaint", ["packag", "box", "arrived"], ["damage", "broken", "crush", "leak", "arriv"]),
    ("What do customers complain about regarding delivery and shipping?", "complaint", ["ship", "deliver", "arrived"], ["late", "slow", "damage", "delay", "arriv"]),
    ("Do customers complain that the products are too expensive?", "complaint", ["price", "expensive", "cost"], ["expensive", "price", "overpriced", "cost"]),
    ("Do customers think the snacks are good value for money?", "praise", ["price", "value", "worth", "cheap"], ["price", "value", "worth", "afford", "cheap"]),
    ("What are the complaints about chocolate?", "complaint", ["chocolate"], ["bitter", "waxy", "taste", "melt", "stale"]),
    ("What do people love about the chocolate?", "praise", ["chocolate"], ["rich", "taste", "delicious", "flavor", "smooth"]),
    ("What do dog owners complain about?", "complaint", ["dog"], ["dog", "eat", "sick", "refus"]),
    ("What do dog owners like about the treats?", "praise", ["dog"], ["dog", "love", "treat", "enjoy"]),
    ("What problems do cat owners report?", "complaint", ["cat"], ["cat", "eat", "refus", "smell"]),
    ("What do people enjoy about the chips and snacks?", "praise", ["chip", "snack"], ["crunch", "taste", "flavor", "delicious"]),
    ("What are the complaints about stale or broken chips?", "complaint", ["chip", "stale", "crunch"], ["stale", "broken", "crush", "taste"]),
    ("What do customers like about healthy or organic products?", "praise", ["healthy", "organic", "natural"], ["healthy", "organic", "natural", "ingredient"]),
    ("Do customers complain that some products are too sweet?", "complaint", ["sweet", "sugar"], ["sweet", "sugar"]),
    ("What do people like about the coffee?", "praise", ["coffee"], ["taste", "smooth", "rich", "flavor", "aroma"]),
    ("What kinds of products do these reviews talk about?", "general", None, ["coffee", "tea", "snack", "food", "chocolate"]),
    ("What complaints are there about expired or stale products?", "complaint", ["expired", "stale", "date"], ["expired", "stale", "old", "date"]),
    ("What do people like about the cookies?", "praise", ["cookie"], ["taste", "delicious", "soft", "flavor"]),
    ("What do customers say about smartphone battery life?", "refusal", None, None),  # off-topic: should be refused
]


def run_one(question: str) -> dict:
    if RUN_LLM:
        return graph.ask(question)
    state = {"question": question}
    state.update(graph.router(state))
    state.update(graph.retrieve(state))
    state.update(graph.grade(state))
    state["answer"] = ""
    return state


rows = []
try:
    for i, (q, exp_route, topic_kw, theme_kw) in enumerate(TESTS, start=1):
        print(f"[{i}/{len(TESTS)}] {q}")
        t0 = time.time()
        try:
            out = run_one(q)
            error = ""
        except Exception as e:
            out, error = {}, str(e)[:300]
        seconds = time.time() - t0

        docs = out.get("docs", [])
        answer = out.get("answer", "")
        route = out.get("route", "")
        relevant = out.get("relevant", False)

        # 1. routing
        route_ok = (route == exp_route) if (exp_route != "refusal" and not error) else None  # API failures are not routing misses

        # 2. retrieval precision: share of retrieved reviews that mention a topic keyword
        precision = None
        if topic_kw and docs:
            hits = sum(
                any(k in (d["text"] + " " + d["summary"]).lower() for k in topic_kw) for d in docs
            )
            precision = hits / len(docs)

        # 3. theme match: answer mentions an expected theme word
        theme_ok = None
        if RUN_LLM and theme_kw and answer:
            theme_ok = any(k in answer.lower() for k in theme_kw)

        # 4. grounded citations: every cited review ID was actually retrieved
        cited = set(re.findall(r"R\d{4}", answer))
        retrieved_ids = {d["id"] for d in docs}
        citations_ok = (cited <= retrieved_ids) if (RUN_LLM and answer) else None

        # 5. off-topic question should be refused (no relevant reviews found)
        refusal_ok = (not relevant) if exp_route == "refusal" else None

        rows.append(
            {
                "#": i,
                "question": q,
                "expected_route": exp_route,
                "route": route,
                "route_ok": route_ok,
                "retrieval_precision": precision,
                "theme_ok": theme_ok,
                "citations_ok": citations_ok,
                "refusal_ok": refusal_ok,
                "seconds": round(seconds, 1),
                "answer": answer,
                "error": error,
                "your_verdict (correct / partial / wrong)": "",
            }
        )
        time.sleep(PAUSE_SECONDS)
except KeyboardInterrupt:
    print("Stopped early, saving what we have...")

df = pd.DataFrame(rows)


def rate(col):
    s = df[col].dropna()
    return round(100 * s.astype(bool).mean(), 1) if len(s) else None


summary = pd.DataFrame(
    {
        "metric": [
            "Questions run",
            "Questions with errors",
            "Routing accuracy (%)",
            "Avg retrieval precision (%)",
            "Theme match rate (%)",
            "Grounded citations (%)",
            "Off-topic question refused (%)",
            "Avg seconds per question",
            "Your manual accuracy (fill the verdict column)",
        ],
        "value": [
            len(df),
            int((df["error"] != "").sum()),
            rate("route_ok"),
            round(100 * df["retrieval_precision"].dropna().mean(), 1) if df["retrieval_precision"].notna().any() else None,
            rate("theme_ok"),
            rate("citations_ok"),
            rate("refusal_ok"),
            round(df["seconds"].mean(), 1),
            '=IFERROR(ROUND(100*(COUNTIF(results!M:M,"correct")+0.5*COUNTIF(results!M:M,"partial"))/COUNTA(results!B2:B100),1),"")',
        ],
    }
)

with pd.ExcelWriter("evaluation_results.xlsx", engine="openpyxl") as w:
    summary.to_excel(w, sheet_name="summary", index=False)
    df.to_excel(w, sheet_name="results", index=False)
    for name, widths in (("summary", {"A": 48, "B": 16}), ("results", {"B": 55, "K": 90, "M": 28})):
        for col, wd in widths.items():
            w.sheets[name].column_dimensions[col].width = wd

print("\n" + summary.to_string(index=False))
print("\nSaved evaluation_results.xlsx")
