"""
Step 2: the agentic RAG flow (LangGraph + Pinecone + Gemini).

Setup:
    pip install langgraph google-genai
    Add this line to your .env file:
        GEMINI_API_KEY=your_gemini_key_here

Run a quick test in the terminal:
    python graph.py
"""
import os
import time
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "reviews"
# Tried in order. Add more names from `python list_models.py` (a "flash-lite" model has a separate, larger free quota).
MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
TOP_K = 20
MAX_RETRIES = 1

# One attempt per model and a 30 second limit, so a busy model fails fast and we move to the next one.
try:
    _http = types.HttpOptions(timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1))
except AttributeError:  # older google-genai versions: at least set the timeout
    _http = types.HttpOptions(timeout=30_000)
llm = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=_http)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)


def chat(system: str, user: str) -> str:
    """Call Gemini, retrying on temporary errors and falling back to a second model."""
    config = types.GenerateContentConfig(system_instruction=system, temperature=0)
    errors = []
    for model in MODELS:
        for attempt in range(1):
            try:
                t0 = time.time()
                resp = llm.models.generate_content(model=model, contents=user, config=config)
                print(f"[{model}] answered in {time.time() - t0:.1f}s")
                return (resp.text or "").strip()
            except Exception as e:
                errors.append(f"[{model}] {str(e)[:200]}")
                if any(code in str(e) for code in ("404", "NOT_FOUND", "429", "RESOURCE_EXHAUSTED")):
                    break  # not available or out of quota: retrying will not help, try the next model
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("All Gemini models failed: " + " | ".join(errors))


class State(TypedDict):
    question: str
    query: str                  # the search query (may be rewritten)
    route: str                  # complaint / praise / general
    docs: List[dict]            # retrieved reviews
    relevant: bool
    retries: int
    answer: str


# ---------- nodes ----------
COMPLAINT_WORDS = ("complain", "problem", "issue", "dislike", "worst", "negative", "hate",
                   "disappoint", "wrong", "poor", "drawback", "bad ")
PRAISE_WORDS = ("like", "love", "great", "good", "worth", "value", "recommend", "enjoy", "favorite", "favourite", "positive", "pros", "praise")


def router(state: State) -> dict:
    q = state["question"].lower()
    complaint = any(w in q for w in COMPLAINT_WORDS)
    praise = any(w in q.replace("dislike", "") for w in PRAISE_WORDS)

    if complaint and not praise:
        route = "complaint"          # obvious case: no LLM call needed
    elif praise and not complaint:
        route = "praise"
    else:                            # unclear: ask the LLM
        label = chat(
            "Classify the user's question about product reviews. Reply with exactly one word: "
            "complaint (asks about problems, negatives, dislikes), "
            "praise (asks about positives, what people like), or "
            "general (anything else).",
            state["question"],
        ).lower()
        route = next((w for w in ("complaint", "praise") if w in label), "general")
    return {"route": route, "query": state["question"], "retries": 0}


def retrieve(state: State) -> dict:
    flt: Optional[dict] = None
    if state["route"] == "complaint":
        flt = {"rating": {"$lte": 2}}
    elif state["route"] == "praise":
        flt = {"rating": {"$gte": 4}}

    vec = embedder.encode([state["query"]], normalize_embeddings=True)[0].tolist()
    res = index.query(vector=vec, top_k=TOP_K, include_metadata=True, filter=flt)
    docs = [
        {
            "id": m["id"],
            "score": m["score"],
            "rating": m["metadata"]["rating"],
            "summary": m["metadata"]["summary"],
            "text": m["metadata"]["review_text"],
        }
        for m in res["matches"]
    ]
    return {"docs": docs}


USE_LLM_GRADER = False   # True = ask Gemini to judge relevance (slower, one extra API call)
MIN_SCORE = 0.35         # average similarity of the top 5 reviews; tune this on your test questions


def grade(state: State) -> dict:
    if USE_LLM_GRADER:
        sample = "\n".join(f"- {d['summary']}: {d['text'][:150]}" for d in state["docs"][:6])
        verdict = chat(
            "You check search results. Reply only 'yes' if the reviews below are relevant to the "
            "question, otherwise reply 'no'.",
            f"Question: {state['question']}\n\nReviews:\n{sample}",
        ).lower()
        return {"relevant": verdict.startswith("yes")}

    top = [d["score"] for d in state["docs"][:5]]
    avg = sum(top) / len(top) if top else 0.0
    return {"relevant": avg >= MIN_SCORE}


def rewrite(state: State) -> dict:
    new_query = chat(
        "Rewrite the question as a short search query that would match product review text. "
        "Reply with the query only.",
        state["question"],
    )
    return {"query": new_query, "retries": state["retries"] + 1}


def generate(state: State) -> dict:
    if not state["relevant"]:
        return {
            "answer": "I couldn't find enough relevant reviews to answer that. "
            "Try rephrasing, or ask about taste, packaging, price, or delivery."
        }
    context = "\n".join(
        f"[{d['id']}] ({d['rating']} stars) {d['summary']}: {d['text']}" for d in state["docs"]
    )
    answer = chat(
        "You analyse customer reviews. Using ONLY the reviews provided, answer the question. "
        "Group the answer into 3-5 themes, say roughly how many of the provided reviews "
        "mention each theme, and cite review IDs in square brackets. "
        "State that counts are based on the retrieved sample, not all reviews. "
        "Formatting: write each theme title as a bold line (**Theme name**), "
        "never use markdown headings (#), and keep the whole answer under 250 words. "
        "If the reviews don't support an answer, say so.",
        f"Question: {state['question']}\n\nReviews:\n{context}",
    )
    return {"answer": answer}


# ---------- graph wiring ----------
def after_grade(state: State) -> str:
    if not state["relevant"] and state["retries"] < MAX_RETRIES:
        return "rewrite"
    return "generate"


def build_graph():
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.set_entry_point("router")
    g.add_edge("router", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


app = build_graph()


def ask(question: str) -> dict:
    return app.invoke({"question": question})


if __name__ == "__main__":
    for q in [
        "What do customers complain about regarding coffee?",
        "What do people like about the tea?",
    ]:
        out = ask(q)
        print(f"\nQ: {q}\nRoute: {out['route']} | Retries: {out['retries']}")
        print(out["answer"])
        print("-" * 60)
