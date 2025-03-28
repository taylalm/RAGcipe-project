import sqlite3
import pandas as pd
import os

def query_ground_truth(ingredient_keywords: str, db_path: str = "../DBScript/recipes_clean.db") -> str:
    """
    Given a semicolon-separated string of ingredient keywords,
    query the recipes_clean.db to find recipes whose 'ingredients' field
    contains all those keywords (AND logic).
    Returns a semicolon-separated string of matching recipe IDs.
    """
    # 1. Split keywords by semicolon
    keywords = [kw.strip() for kw in ingredient_keywords.split(";") if kw.strip()]
    if not keywords:
        return ""
    
    conn = sqlite3.connect(db_path)
    
    # 2. Build an AND clause for each keyword
    #    e.g. "ingredients LIKE ? AND ingredients LIKE ?"
    where_clause = " AND ".join(["ingredients LIKE ?"] * len(keywords))
    params = [f"%{kw}%" for kw in keywords]
    
    # 3. Execute the query
    query = f"SELECT id FROM recipes WHERE {where_clause}"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # 4. Combine IDs into a semicolon-separated string
    candidate_ids = ";".join(df["id"].astype(str).tolist())
    return candidate_ids

def main():
    # 1. Load your CSV (e.g. 'Evaluation_Dataset_Recipes.csv') which has at least:
    #    - 'ingredient_keywords' column
    eval_csv = "Evaluation_Dataset_Recipes.csv"
    df = pd.read_csv(eval_csv)
    
    # 2. For each row, call 'query_ground_truth' using the 'ingredient_keywords' field
    df["ground_truth_ids"] = df["ingredient_keywords"].apply(query_ground_truth)
    
    # 3. Save updated CSV
    output_csv = "Evaluation_Dataset_Recipes_with_generated_ground_truth.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Ground truth recipe IDs generated and saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
