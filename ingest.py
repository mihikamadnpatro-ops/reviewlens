"""
Step 1: embed the 5,000 reviews and upload them to Pinecone.

Setup:
    pip install pandas sentence-transformers pinecone python-dotenv
    Create a .env file next to this script containing:
        PINECONE_API_KEY=your_key_here

Run:
    python ingest.py
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "reviews"
CSV_PATH = "reviews_sample_5000.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # produces 384-dimension vectors
BATCH_SIZE = 100

# 1. Load data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} reviews")

# 2. Connect to Pinecone and create the index if it does not exist
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
index = pc.Index(INDEX_NAME)

# 3. Embed and upload in batches
model = SentenceTransformer(MODEL_NAME)

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start : start + BATCH_SIZE]
    vectors = model.encode(batch["combined_text"].tolist(), normalize_embeddings=True)

    records = []
    for (_, row), vec in zip(batch.iterrows(), vectors):
        records.append(
            {
                "id": row["review_id"],
                "values": vec.tolist(),
                "metadata": {
                    "product_id": row["product_id"],
                    "rating": int(row["rating"]),
                    "sentiment": row["sentiment"],
                    "year": int(row["year"]),
                    "summary": row["summary"],
                    "review_text": row["review_text"],
                },
            }
        )
    index.upsert(vectors=records)
    print(f"Uploaded {min(start + BATCH_SIZE, len(df))}/{len(df)}")

# 4. Sanity check: a test query that only looks at negative reviews
query = "the coffee tasted bitter and stale"
q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
res = index.query(
    vector=q_vec,
    top_k=3,
    include_metadata=True,
    filter={"rating": {"$lte": 2}},
)
print(f"\nTest query: {query}")
for m in res["matches"]:
    print(f"- score={m['score']:.2f} rating={m['metadata']['rating']} | {m['metadata']['summary']}")
