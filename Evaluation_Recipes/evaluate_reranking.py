import pandas as pd
import re
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from openai import OpenAI

# --- Set up API key and clients ---
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize cross-encoder for reranking
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ChromaDB setup for recipes (assume embeddings already exist)
chroma_db_path = "../DBScript/chroma_db"  # Adjust relative path if needed
recipes_client = chromadb.PersistentClient(path=chroma_db_path)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-ada-002"
)
recipes_collection = recipes_client.get_or_create_collection(
    name="recipes_collection",
    embedding_function=openai_ef
)

def semantic_search_chroma(query: str, n_results: int = 30) -> List[Dict]:
    """
    Perform a semantic search using ChromaDB.
    Returns a list of candidate recipes as dicts with keys:
       'id', 'name', 'url' (if available), and 'doc' (combined text).
    """
    results = recipes_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas']  # No "ids"
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]

    candidates = []
    for doc, meta in zip(docs, metas):
        # Retrieve ID from meta (assuming it's stored in metadata)
        recipe_id = str(meta.get("id", ""))
        candidates.append({
            "id": recipe_id,
            "name": meta.get("name", "No Name"),
            "url": meta.get("url", "N/A"),
            "doc": doc
        })
    return candidates

def cross_encoder_rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Use the cross-encoder to re-rank candidate recipes.
    Returns the top_k candidates (as dicts).
    """
    if not candidates:
        return []
    pairs = [(query, candidate.get("doc", candidate["name"])) for candidate in candidates]
    scores = cross_encoder_model.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["score"] = score
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return sorted_candidates[:top_k]

def get_recipe_choices(query: str, top_k: int = 5) -> List[Dict]:
    """
    Simplified retrieval pipeline:
      1. Perform semantic search to get top 30 candidates.
      2. Re-rank using the cross-encoder.
      3. Return the top_k candidates.
    """
    semantic_candidates = semantic_search_chroma(query, n_results=30)
    top_candidates = cross_encoder_rerank(query, semantic_candidates, top_k=top_k)
    return top_candidates

# ---------- Evaluation Functions ----------

def precision_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    return len(set(predicted[:k]) & set(ground_truth)) / k

def recall_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    return len(set(predicted[:k]) & set(ground_truth)) / len(ground_truth) if ground_truth else 0

def hits_at_k(predicted: List[str], ground_truth: List[str], k: int) -> int:
    return 1 if len(set(predicted[:k]) & set(ground_truth)) > 0 else 0

def parse_ground_truth_ids(raw_str) -> List[str]:
    if pd.isna(raw_str) or not str(raw_str).strip():
        return []
    return [id_.strip() for id_ in str(raw_str).split(";")]

def evaluate_prompts(evaluation_csv: str, top_k: int = 5) -> None:
    """
    Load an evaluation dataset CSV, run retrieval for each prompt,
    and compute simple metrics (Precision@K, Recall@K, Hits@K).
    
    The CSV should have at least:
       - prompt: the user query
       - ground_truth_ids: semicolon-separated recipe IDs that are correct.
    
    Results are saved to 'Evaluation_Rerank_Results.csv'.
    """
    df = pd.read_csv(evaluation_csv)
    results_list = []
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        ground_truth = parse_ground_truth_ids(row.get('ground_truth_ids'))
        retrieved = get_recipe_choices(prompt, top_k=top_k)
        retrieved_ids = [str(r["id"]) for r in retrieved]
        
        precision = precision_at_k(retrieved_ids, ground_truth, top_k)
        recall = recall_at_k(retrieved_ids, ground_truth, top_k)
        hit = hits_at_k(retrieved_ids, ground_truth, top_k)
        
        results_list.append({
            "prompt": prompt,
            "retrieved_ids": ";".join(retrieved_ids),
            "ground_truth_ids": ";".join(ground_truth),
            "precision@{}".format(top_k): round(precision, 3),
            "recall@{}".format(top_k): round(recall, 3),
            "hit@{}".format(top_k): hit
        })
    
    eval_df = pd.DataFrame(results_list)
    eval_df.to_csv("Evaluation_Rerank_Results1.csv", index=False)
    print("✅ Evaluation completed. Results saved to 'Evaluation_Rerank_Results1.csv'.")

# ---------- Main for Testing One Prompt ----------

if __name__ == "__main__":
    # For quick testing, run one prompt:
    test_prompt = "I want to use eggs,fish and tomatoes in a meal"
    top_recipes = get_recipe_choices(test_prompt, top_k=5)
    print("🔍 Top Recipe Results for prompt:", test_prompt)
    for r in top_recipes:
        print(f"- ID: {r['id']} | Name: {r['name']} | Score: {r['score']:.3f}")
    
    # Run full evaluation on your CSV (uncomment below when ready):
    # evaluate_prompts("Evaluation_Dataset_Recipes_Final.csv", top_k=1)
