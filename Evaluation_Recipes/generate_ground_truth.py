import pandas as pd
import sqlite3
import os
import re

def load_recipes_db(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT id, name, ingredients, calories, protein, fat, carbohydrates
        FROM recipes_clean
    """, conn)
    conn.close()
    return df

def match_recipes(row, recipes_df):
    terms = str(row['ingredient_keywords']).lower().split(";") if pd.notna(row['ingredient_keywords']) else []
    calorie_min = row.get('calorie_min')
    protein_min = row.get('protein_min')
    fat_max = row.get('fat_max')
    carb_max = row.get('card_max')

    matches = recipes_df.copy()

    # Ingredient matching using AND logic with plural-safe word boundaries
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term.strip())}s?\b", re.IGNORECASE)
        matches = matches[matches['ingredients'].apply(lambda x: bool(pattern.search(str(x))))]

    # Nutrition filters
    if pd.notna(calorie_min):
        matches = matches[matches['calories'] >= calorie_min]
    if pd.notna(protein_min):
        matches = matches[matches['protein'] >= protein_min]
    if pd.notna(fat_max):
        matches = matches[matches['fat'] <= fat_max]
    if pd.notna(carb_max):
        matches = matches[matches['carbohydrates'] <= carb_max]

    return matches[['id', 'name']]

def main():
    # File paths
    EVAL_PATH = "Evaluation_Dataset_Recipes.csv"
    DB_PATH = "../DBScript/recipes_clean.db"  # Update if path differs

    # Load data
    eval_df = pd.read_csv(EVAL_PATH)
    recipes_df = load_recipes_db(DB_PATH)

    # Create ground truth columns
    ground_names = []
    ground_ids = []

    for _, row in eval_df.iterrows():
        matches = match_recipes(row, recipes_df)
        ground_names.append(";".join(matches['name'].tolist()))
        ground_ids.append(";".join(matches['id'].astype(str).tolist()))

    # Add back to dataframe and save
    eval_df['ground_truth_names'] = ground_names
    eval_df['ground_truth_ids'] = ground_ids
    eval_df.to_csv(EVAL_PATH, index=False)

    print(f"✅ Ground truth saved to: {EVAL_PATH}")

if __name__ == "__main__":
    main()
