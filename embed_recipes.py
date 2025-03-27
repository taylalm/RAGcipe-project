import sqlite3
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import openai
import os
import tiktoken

# 1️⃣ Set up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure API key is set in .env or environment variable

# 2️⃣ Connect to SQLite database and load data
conn = sqlite3.connect("recipes.db")
df = pd.read_sql_query("SELECT * FROM recipes", conn)
conn.close()

print(f"✅ Loaded {len(df)} recipes from SQLite.")

# 3️⃣ Combine fields into a single text block for embedding
df['combined_text'] = (
    "Recipe Name: " + df['name'].fillna('') + "\n"
    "Ingredients: " + df['ingredients'].fillna('') + "\n"
    "Method: " + df['method'].fillna('') + "\n"
    "Nutritional Info: " + df['nutritional_data'].fillna('')
)

print("✅ Combined text fields ready.")

# 4️⃣ Count token length per recipe
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
df['token_count'] = df['combined_text'].apply(lambda x: len(encoding.encode(x)))

# Display how many exceed token limit
high_token_recipes = df[df['token_count'] > 1000]
print(f"⚠️ Skipping {len(high_token_recipes)} recipes with token_count > 1000")

# Save excluded recipes for reference
high_token_recipes[['id', 'name', 'token_count']].to_csv("excluded_recipes.csv", index=False)

# Filter out high-token recipes
filtered_df = df[df['token_count'] <= 1000].reset_index(drop=True)
print(f"✅ Proceeding to embed {len(filtered_df)} recipes...")

# Optional: Save list of recipes being embedded
filtered_df[['id', 'name', 'token_count']].to_csv("embedded_recipes.csv", index=False)

# 5️⃣ Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient(path="chroma_db")
print("✅ ChromaDB Persistent Client initialized.")

# 6️⃣ Set up OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)

# 7️⃣ Create or retrieve ChromaDB collection
collection = client.get_or_create_collection(
    name="recipes_collection",
    embedding_function=openai_ef
)
print("✅ ChromaDB Collection created/retrieved.")

# 8️⃣ Add documents in manageable batches
batch_size = 50
print(f"🔄 Adding {len(filtered_df)} documents to ChromaDB in batches of {batch_size}...")

for start_idx in range(0, len(filtered_df), batch_size):
    end_idx = start_idx + batch_size
    batch_df = filtered_df.iloc[start_idx:end_idx]

    documents = batch_df['combined_text'].tolist()
    ids = batch_df['id'].astype(str).tolist()
    metadata = batch_df[['name', 'url']].to_dict(orient='records')

    collection.add(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    print(f"✅ Batch {start_idx // batch_size + 1} added")

# 9️⃣ Persist ChromaDB
client.persist()
print("🎉 Embedding completed and saved to ChromaDB!")
print("📄 Excluded recipes saved to: excluded_recipes.csv")
print("📄 Embedded recipes saved to: embedded_recipes.csv")
