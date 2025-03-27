import chromadb
from chromadb.utils import embedding_functions
import sqlite3
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-ada-002"
)

# ChromaDB client setup
client = chromadb.PersistentClient(path="fairprice_openai_embeddings_db")
product_collection = client.get_or_create_collection(
    "fairprice_products_openai", embedding_function=openai_ef
)

# Load selected data from SQLite
conn = sqlite3.connect("ingredient_chroma_db/fairprice_items.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT id, name, brand, category, key_information, additional_information, 
           ingredients, dietary, origin, nutritional_data, price, size, ratings, url
    FROM products
""")

products = cursor.fetchall()

# Embed with ideal semantic fields, clearly store metadata
# Embed with ideal semantic fields, handling None clearly
for product in products:
    (pid, name, brand, category, key_info, add_info, 
     ingredients, dietary, origin, nutrition, price, size, ratings, url) = product

    # Replace None clearly for embeddings
    name = name or ""
    brand = brand or ""
    category = category or ""
    key_info = key_info or ""
    add_info = add_info or ""
    ingredients = ingredients or ""
    dietary = dietary or ""
    origin = origin or ""
    nutrition = nutrition or ""

    embedding_text = (
        f"{name} by {brand}. Category: {category}. {key_info}. Ingredients: {ingredients}. "
        f"Additional info: {add_info}. Dietary: {dietary}. Origin: {origin}. Nutrition: {nutrition}."
    )

    # Replace None values in metadata clearly
    metadata = {
        "name": name,
        "brand": brand,
        "category": category,
        "price": price if price is not None else -1,
        "size": size or "Not specified",
        "ratings": ratings if ratings is not None else -1,
        "url": url or ""
    }

    product_collection.add(
        ids=[str(pid)],
        documents=[embedding_text],
        metadatas=[metadata]
    )

print("✅ Successfully embedded products with metadata clearly handling None values.")
