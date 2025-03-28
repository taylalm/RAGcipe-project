import sqlite3
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import openai
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

# ========================
# STEP 3: EMBED RECIPES
# ========================

def embed_recipes():
    """Embed recipes from recipes_clean.db using OpenAI and save to ChromaDB"""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    conn = sqlite3.connect("recipes_clean.db")
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

    # Filter to recipes within token limit
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
# MAIN
# ========================

if __name__ == "__main__":
    embed_recipes()
