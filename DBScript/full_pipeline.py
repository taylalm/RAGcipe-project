import sqlite3
import re
import csv
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import openai
import tiktoken
from dotenv import load_dotenv
load_dotenv()


# ========================
# STEP 1: CLEAN RECIPES
# ========================

def clean_recipes():
    """Remove junk recipes, empty names, and fix quotes. Save cleaned DB and CSV."""
    conn = sqlite3.connect("recipes.db")
    df = pd.read_sql_query("SELECT * FROM recipes", conn)
    conn.close()

    # 1. Remove rows with blank or null names
    df = df[df['name'].notna() & df['name'].str.strip().ne("")]

    # 2. Remove promo recipes like "Lower in Sodium..." or "HEALTHIER CHOICE"
    df = df[~df['name'].str.contains("HEALTHIER|Lower in Sodium", case=False, na=False)]

    # 3. Remove rows where name is exactly "Tip"
    df = df[~df['name'].str.strip().str.lower().eq("tip")]

    # 4. Remove quotes in recipe names
    df['name'] = df['name'].str.replace('"', '', regex=False).str.strip()

    # ✅ Save to CSV for inspection
    df[['id', 'name']].to_csv("recipes_before_emb.csv", index=False)

    # Save cleaned recipes to SQLite
    cleaned_conn = sqlite3.connect("recipes_cleaned.db")
    df.to_sql("recipes", cleaned_conn, index=False, if_exists="replace")
    cleaned_conn.close()

    print(f"✅ Cleaned recipes saved to 'recipes_cleaned.db' and 'recipes_before_emb.csv' ({len(df)} rows)")



# ========================
# STEP 2: TOKEN COUNT
# ========================

def check_token_lengths():
    """Check token count for recipes in recipes_cleaned.db and save high/low token lists."""
    conn = sqlite3.connect("recipes_cleaned.db")
    df = pd.read_sql_query("SELECT * FROM recipes", conn)
    conn.close()

    df['combined_text'] = (
        "Recipe Name: " + df['name'].fillna('') + "\n"
        "Ingredients: " + df['ingredients'].fillna('') + "\n"
        "Method: " + df['method'].fillna('') + "\n"
        "Nutritional Info: " + df['nutritional_data'].fillna('')
    )

    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    df['token_count'] = df['combined_text'].apply(lambda x: len(encoding.encode(x)))

    df[['id', 'name', 'token_count']].to_csv("token_counts.csv", index=False)
    df[df['token_count'] > 1000][['id', 'name', 'token_count']].to_csv("high_token_recipes.csv", index=False)

    print(f"✅ Token counts saved. High token recipes: {len(df[df['token_count'] > 1000])}")


# ========================
# STEP 3: EMBED RECIPES
# ========================

def embed_recipes():
    """Embed recipes from recipes_cleaned.db using OpenAI and save to ChromaDB"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    conn = sqlite3.connect("recipes_cleaned.db")
    df = pd.read_sql_query("SELECT * FROM recipes", conn)
    conn.close()

    df['combined_text'] = (
        "Recipe Name: " + df['name'].fillna('') + "\n"
        "Ingredients: " + df['ingredients'].fillna('') + "\n"
        "Method: " + df['method'].fillna('') + "\n"
        "Nutritional Info: " + df['nutritional_data'].fillna('')
    )

    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    df['token_count'] = df['combined_text'].apply(lambda x: len(encoding.encode(x)))

    # Filter
    filtered_df = df[df['token_count'] <= 1000].reset_index(drop=True)
    filtered_df[['id', 'name', 'token_count']].to_csv("embedded_recipes.csv", index=False)

    # Embed
    client = chromadb.PersistentClient(path="chroma_db")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )
    collection = client.get_or_create_collection(
        name="recipes_collection",
        embedding_function=openai_ef
    )

    batch_size = 50
    for start_idx in range(0, len(filtered_df), batch_size):
        end_idx = start_idx + batch_size
        batch_df = filtered_df.iloc[start_idx:end_idx]
        documents = batch_df['combined_text'].tolist()
        ids = batch_df['id'].astype(str).tolist()
        metadata = batch_df[['name', 'url']].to_dict(orient='records')
        collection.add(documents=documents, metadatas=metadata, ids=ids)
        print(f"✅ Batch {start_idx // batch_size + 1} embedded.")

    print(f"✅ {len(filtered_df)} recipes embedded and saved to ChromaDB.")


# ========================
# STEP 4: CREATE FINAL DB
# ========================

def create_filtered_sqlite_db():
    """Create recipes_final.db using only recipes that were embedded"""
    embedded = pd.read_csv("embedded_recipes.csv")
    embedded_ids = tuple(embedded["id"].tolist())

    conn = sqlite3.connect("recipes_cleaned.db")
    query = f"SELECT * FROM recipes WHERE id IN {embedded_ids}"
    df_filtered = pd.read_sql_query(query, conn)
    conn.close()

    final_conn = sqlite3.connect("recipes_final.db")
    df_filtered.to_sql("recipes", final_conn, index=False, if_exists="replace")
    final_conn.close()

    print(f"✅ Final DB created: recipes_final.db ({len(df_filtered)} recipes)")


# ========================
# STEP 5: PARSE NUTRITION
# ========================


def clean_energy_line(nutri_text):
    """Handle special Energy (1 kcal = 4.2kJ) ... kcal format."""
    # Match both '1kcal' and '1 kcal'
    energy_match = re.search(r"Energy\s*\(1\s*kcal\s*=\s*4\.2kJ\)\s*(\d+(?:\.\d+)?)\s*k?cal", nutri_text, re.IGNORECASE)
    if energy_match:
        value = energy_match.group(1)
        # Replace the whole thing with Energy: <value> kcal
        nutri_text = re.sub(r"Energy\s*\(1\s*kcal\s*=\s*4\.2kJ\)\s*\d+(?:\.\d+)?\s*k?cal", f"Energy: {value} kcal", nutri_text, flags=re.IGNORECASE)
    return nutri_text

def parse_nutrition_data(nutri_text):
    """Extract numerical nutrition values from messy strings."""
    data = {
        "calories": None,
        "protein": None,
        "fat": None,
        "cholesterol": None,
        "carbohydrates": None,
        "fibre": None,
        "sodium": None
    }

    if not nutri_text or nutri_text.strip().lower() in ["not available", "phone lines are open"]:
        return data

    # Fix energy line first
    nutri_text = clean_energy_line(nutri_text)

    # Remove junk like (g and %...)
    nutri_text = re.sub(r"\(g\s+and\s+%[^)]*\)", "", nutri_text, flags=re.IGNORECASE)

    patterns = {
        "calories": re.compile(r"(?:Energy|Calories)[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*k?cal", re.IGNORECASE),
        "protein": re.compile(r"Protein[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*g", re.IGNORECASE),
        "fat": re.compile(r"(?:Total\s+)?Fat[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*g", re.IGNORECASE),
        "cholesterol": re.compile(r"Cholesterol[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE),
        "carbohydrates": re.compile(r"(?:Carbohydrates?|Carbohydrate)[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*g", re.IGNORECASE),
        "fibre": re.compile(r"(?:Fibre|Fiber|Dietary\s+Fibre)[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*g", re.IGNORECASE),
        "sodium": re.compile(r"Sodium[^:\d]*[:=]?\s*(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE)
    }

    for key, pattern in patterns.items():
        match = pattern.search(nutri_text)
        if match:
            try:
                data[key] = float(match.group(1))
            except:
                data[key] = None

    return data

def parse_nutrition():
    """Parse nutrition text and save to recipes_clean.db"""
    conn = sqlite3.connect("recipes_final.db")
    df = pd.read_sql_query("SELECT * FROM recipes", conn)
    conn.close()

    print(f"📥 Loaded {len(df)} rows from recipes_final.db")

    parsed_rows = []
    for _, row in df.iterrows():
        parsed = parse_nutrition_data(row['nutritional_data'])
        parsed_rows.append({**row, **parsed})

    df_parsed = pd.DataFrame(parsed_rows)

    print(f"📤 Saving {len(df_parsed)} rows to recipes_clean.db")

    conn_clean = sqlite3.connect("recipes_clean.db")
    df_parsed.to_sql("recipes_clean", conn_clean, index=False, if_exists='replace')
    conn_clean.close()

    print(f"✅ Parsed nutrition and saved to recipes_clean.db")



# ========================
# RUN SELECTED STEP
# ========================

if __name__ == "__main__":
    # Uncomment what you want to run:

    # clean_recipes()
    # check_token_lengths()
    # embed_recipes()
    # create_filtered_sqlite_db()
    parse_nutrition()
    pass
