import sqlite3
import re
import pandas as pd
from typing import List, Dict
import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from openai import OpenAI

# Load API key
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Cross-encoder for reranking
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Chroma Setup
recipes_client = chromadb.PersistentClient(path="../DBScript/chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-ada-002"
)
recipes_collection = recipes_client.get_collection("recipes_collection", embedding_function=openai_ef)

NUTRI_THRESHOLDS = {
    "calorie": ("calories", 250),
    "protein": ("protein", 20),
    "fat": ("fat", 8),
    "cholesterol": ("cholesterol", 60),
    "carbohydrate": ("carbohydrates", 40),
    "fibre": ("fibre", 5),
    "fiber": ("fibre", 5),
    "sodium": ("sodium", 400),
}

def detect_nutrition_filters(prompt: str) -> dict:
    filters = {}
    prompt_lower = prompt.lower()
    for word, (col, threshold) in NUTRI_THRESHOLDS.items():
        if f"low {word}" in prompt_lower:
            filters[col] = ("<", threshold)
        elif f"high {word}" in prompt_lower:
            filters[col] = (">=", threshold)
    return filters

def semantic_search_chroma(query: str, n_results: int = 30) -> List[Dict]:
    results = recipes_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas']
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]

    # Build list
    out = []
    for doc, meta in zip(docs, metas):
        out.append({
            "id": str(meta.get("id", "")),
            "name": meta.get("name", "No Name"),
            "url": meta.get("url", "N/A"),
            "doc": doc
        })
    return out

def sql_filter_by_ids(candidate_ids: List[str]) -> pd.DataFrame:
    if not candidate_ids:
        return pd.DataFrame()  # empty
    conn = sqlite3.connect("../DBScript/recipes_clean.db")
    placeholders = ",".join("?" for _ in candidate_ids)
    query = f"""
        SELECT id, name, url, ingredients, calories, protein, fat, cholesterol, carbohydrates, fibre, sodium
        FROM recipes_clean
        WHERE id IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=candidate_ids)
    conn.close()
    return df

def cross_encoder_rerank(query: str, candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    pairs = [(query, row["ingredients"]) for _, row in candidates.iterrows()]
    scores = cross_encoder_model.predict(pairs)
    candidates["score"] = scores
    # sort descending by score
    candidates = candidates.sort_values("score", ascending=False)
    return candidates.head(top_k)

def get_recipe_choices_hybrid(prompt: str, top_k: int = 5) -> pd.DataFrame:
    """ 
    1) Chroma semantic search
    2) SQL join by ID
    3) Apply nutrition filters
    4) Cross-encoder rerank
    5) Return top_k
    """
    print(f"🔎 Prompt: {prompt}")

    # Step 1: Chroma
    chroma_candidates = semantic_search_chroma(prompt, n_results=30)
    print(f"💠 Chroma found {len(chroma_candidates)} candidates.")
    for i, c in enumerate(chroma_candidates[:5]):
        print(f"   Chroma cand {i+1}: ID={c['id']} Name={c['name'][:30]}...")

    candidate_ids = [c["id"] for c in chroma_candidates if c["id"]]
    print(f"🆔 Candidate IDs: {candidate_ids}")

    # Step 2: SQL join
    candidates_df = sql_filter_by_ids(candidate_ids)
    print(f"📊 SQL join returned {len(candidates_df)} rows.")
    if not candidates_df.empty:
        print(candidates_df.head())

    # Step 3: Nutrition filters
    filters = detect_nutrition_filters(prompt)
    print(f"🔍 Detected filters: {filters}")
    for col, (op, threshold) in filters.items():
        if col in candidates_df.columns:
            if op == "<":
                before_count = len(candidates_df)
                candidates_df = candidates_df[candidates_df[col] < threshold]
                print(f"   Filter {col} < {threshold}: {before_count} -> {len(candidates_df)} remain")
            elif op == ">=":
                before_count = len(candidates_df)
                candidates_df = candidates_df[candidates_df[col] >= threshold]
                print(f"   Filter {col} >= {threshold}: {before_count} -> {len(candidates_df)} remain")

    if candidates_df.empty:
        print("⚠️ No candidates left after filtering.")
        return candidates_df

    # Step 4: Cross-encoder rerank
    reranked_df = cross_encoder_rerank(prompt, candidates_df, top_k=top_k)
    print(f"⭐ Reranked top {top_k}: {len(reranked_df)} found")
    print(reranked_df[["id", "name", "score"]])

    return reranked_df

# Test one prompt
if __name__ == "__main__":
    test_prompt = "What dessert can I make with apples and honey?"
    final_df = get_recipe_choices_hybrid(test_prompt, top_k=5)

    print("\n✅ Final top recipes:")
    if not final_df.empty:
        for idx, row in final_df.iterrows():
            print(f"- ID={row['id']} Name={row['name']} Score={row['score']}")
    else:
        print("No results found.")
