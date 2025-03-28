import sqlite3
import re
import pandas as pd
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Nutrition thresholds
NUTRI_THRESHOLDS = {
    "calorie": ("calories", 250),
    "protein": ("protein", 20),
    "fat": ("fat", 8),
    "cholesterol": ("cholesterol", 60),
    "carbohydrate": ("carbohydrates", 40),
    "fibre": ("fibre", 5),
    "fiber": ("fibre", 5),  # US spelling
    "sodium": ("sodium", 400),
}

def detect_nutrition_filters(prompt: str) -> dict:
    """Detect vague nutrition terms (e.g., low fat) in the prompt."""
    filters = {}
    for word, (col, threshold) in NUTRI_THRESHOLDS.items():
        if f"low {word}" in prompt.lower():
            filters[col] = ("<", threshold)
        elif f"high {word}" in prompt.lower():
            filters[col] = (">=", threshold)
    return filters

def get_filtered_recipes_from_sql(ingredients: List[str], nutrition_filters: dict) -> pd.DataFrame:
    """Filter recipes by ingredients and nutrition thresholds."""
    conn = sqlite3.connect("../DBScript/recipes_clean.db")
    query = """
        SELECT id, name, url, ingredients, calories, protein, fat, cholesterol, carbohydrates, fibre, sodium
        FROM recipes_clean
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Ingredient filtering
    if ingredients:
        for ingredient in ingredients:
            pattern = rf"\b{ingredient.strip()}s?\b"  # Handle plural (apple → apples)
            df = df[df['ingredients'].str.contains(pattern, case=False, na=False, regex=True)]

    # Nutrition filtering
    for col, (op, val) in nutrition_filters.items():
        if op == "<":
            df = df[df[col] < val]
        elif op == ">=":
            df = df[df[col] >= val]

    return df.reset_index(drop=True)

def rerank_with_llm(prompt: str, candidates: List[str]) -> List[int]:
    """Use LLM to rerank a list of candidate recipe names."""
    if not candidates:
        return []

    system_msg = "You are a helpful assistant that selects the most relevant recipes."
    examples = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    user_msg = (
        f"User query:\n\"{prompt}\"\n\n"
        f"Candidate recipes:\n{examples}\n\n"
        f"Please rank the top recipes in descending order using just the numbers (e.g. 5,2,1)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        ranked_indices = re.findall(r"\d+", response.choices[0].message.content)
        return [int(i) - 1 for i in ranked_indices if int(i) - 1 < len(candidates)]
    except Exception as e:
        print("⚠️ LLM reranking failed:", e)
        return list(range(len(candidates)))  # fallback to default order

def get_recipe_choices(prompt: str, top_k: int = 5, mode: str = "user", ingredient_keywords: List[str] = []) -> List[Dict]:
    """
    Given a prompt and optional ingredients, return top-ranked recipes.
    Returns: List of dicts with 'id', 'name', and optional 'url'
    """
    nutrition_filters = detect_nutrition_filters(prompt)
    filtered_df = get_filtered_recipes_from_sql(ingredient_keywords, nutrition_filters)

    if filtered_df.empty:
        return []

    if mode == "eval":
        # Use head to slice the DataFrame
        return [{"id": row["id"], "name": row["name"]} 
                for _, row in filtered_df.head(top_k).iterrows()]

    # Rerank with LLM
    candidates = filtered_df['name'].tolist()
    ranked_indices = rerank_with_llm(prompt, candidates)

    results = []
    for i in ranked_indices[:top_k]:
        row = filtered_df.iloc[i]
        result = {"id": row["id"], "name": row["name"]}
        if "url" in filtered_df.columns and not pd.isna(row["url"]):
            result["url"] = row["url"]
        results.append(result)
    return results
