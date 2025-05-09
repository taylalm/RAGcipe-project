# TEST.py

import sys
try:
    import pysqlite3  # This is the pip-installed "pysqlite3-binary" package
    # Re-map the built-in "sqlite3" to "pysqlite3"
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
import requests
from functools import lru_cache

# --- URL Validation with Caching and Retry ---
@lru_cache(maxsize=1000)
def is_valid_url(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# --- Load Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# --- Initialize Cross-Encoder for Reranking ---
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, metadatas, top_k=5):
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder_model.predict(pairs)
    ranked_results = sorted(
        zip(documents, metadatas, scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    return ranked_results[:top_k]

# --- ChromaDB Setup for Recipes ---
recipes_client = chromadb.PersistentClient(path="chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, 
    model_name="text-embedding-ada-002"
)
recipes_collection = recipes_client.get_collection(
    "recipes_collection", 
    embedding_function=openai_ef
)

# --- ChromaDB Setup for Ingredients (FairPrice) ---
ingredients_client = chromadb.PersistentClient(path="fairprice_openai_embeddings_db")
ingredients_collection = ingredients_client.get_collection(
    "fairprice_products_openai", 
    embedding_function=openai_ef
)

def generate_prompt(user_query, recipe_name, recipe_url, recipe_details, nutritional_data, ingredients_from_db):
    ingredient_str = ""
    for ing, products in ingredients_from_db.items():
        ingredient_str += f"\n**{ing.capitalize()}** (Price details provided):\n"
        for prod in products:
            meta = prod['metadata']
            product_url = meta.get('url', 'N/A')
            if product_url != 'N/A' and not is_valid_url(product_url):
                continue
            ingredient_str += (
                f"- {meta['name']} by {meta['brand']} "
                f"(Price: ${meta['price']}, Size: {meta['size']}, URL: {product_url})\n"
            )
    
    prompt = f"""
You are an expert culinary assistant.

A user is seeking recipe suggestions for the query: "**{user_query}**". 
In addition to providing a detailed recipe summary, your task is to help the user make an affordable, healthy purchase by:
1. Analyzing the available FairPrice ingredient options and suggesting suitable ingredient substitutions clearly if any.
2. For each necessary ingredient, among the multiple product options provided, identifying the three most relevant and cost-effective products (based on price and quantity) including their price, source URL and quantity.
3. Providing nutritional information clearly based on the provided nutritional data.
4. Optionally estimating the total cost of the required ingredients.

Below is the retrieved recipe and a list of FairPrice ingredient products with their price and source URL information. Please include the source URL for the recipe and each product options in your response for clarity and reliability.

---

**Retrieved Recipe:**
- **Recipe Name:** {recipe_name}
- **URL:** {recipe_url}
- **Details:**
{recipe_details}

**Nutritional Information:**
{nutritional_data}

---

**FairPrice Ingredient Products:**
{ingredient_str}

---

Please provide your response in four sections:
1. **Recipe Summary** – Summarize the key steps and ingredients in a concise and clear paragraph, including the recipe source URL.
2. **Affordable Ingredient Recommendations** – For each necessary ingredient, identify the three most relevant and cost-effective FairPrice products (based on price and quantity), including their price, source URL and quantity.
3. **Nutritional Analysis** – Provide a clear analysis based on the nutritional information of the recipe and the ingredients. Discuss the health benefits or potential dietary advantages (e.g., high protein content, low saturated fat, rich in fiber, etc.). Mention who might benefit from this dish (e.g., vegetarians, fitness enthusiasts, people watching cholesterol).
4. **Cost Estimate**: 
- Estimate the total cost to prepare this recipe using the selected FairPrice ingredients.
- If an ingredient has multiple product options, select the most relevant, cost-effective combination (based on price and quantity) to estimate the total cost. Relevancy is more important than cost efficiency.
- For each product in the chosen combination, include its price, quantity purchased, and URL.
- Determine how many full servings can be made with the purchased quantities based on the recipe’s required amount of each ingredient.
- If the initial estimate results in only 1 serving due to a limiting ingredient, suggest whether it’s reasonable to purchase more of that ingredient to increase the number of servings and lower the cost per serving.
- Provide both:
+ The cost per serving based on the original ingredient purchase
+ An optimized cost per serving assuming the user buys more of the limiting ingredient (if it leads to better cost-efficiency).
+ Break down how much each ingredient contributes to the cost of a single serving.
"""
    return prompt

def get_llm_response(prompt, model="gpt-4o", temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def extract_ingredients(recipe_text):
    match = re.search(
        r'Ingredients:(.*?)(Method|Nutritional Info)', 
        recipe_text, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        ingredients_block = match.group(1).strip()
        ingredients_list = [
            re.sub(r'[\d\*\(\),]+', '', line).strip().lower()
            for line in ingredients_block.split('\n') 
            if line.strip()
        ]
        return list(set(ingredients_list))
    return []

def search_ingredients_chroma(ingredient_name, desired=3):
    if not ingredient_name or not ingredient_name.strip():
        return []
    try:
        results = ingredients_collection.query(
            query_texts=[ingredient_name],
            n_results=10,
            include=['metadatas', 'documents', 'distances']
        )
        matched_products = []
        for meta, doc, dist in zip(results['metadatas'][0], 
                                   results['documents'][0], 
                                   results['distances'][0]):
            matched_products.append({
                "metadata": meta, 
                "document": doc, 
                "similarity": dist
            })
        return matched_products[:desired]
    except Exception as e:
        print(f"Error querying ingredient '{ingredient_name}': {e}")
        return []

def boost_recipes(query_text, recipes):
    boosting_keywords = ["low carb", "halal", "iron rich", "low sodium"]
    query_lower = query_text.lower()
    for recipe in recipes:
        combined_text = (recipe["name"] + " " + recipe["document"]).lower()
        boost = 0
        for keyword in boosting_keywords:
            if keyword in query_lower and keyword in combined_text:
                boost += 1
        recipe["boost"] = boost
    return recipes

def get_recipe_choices(query_text, n_results=3):
    recipe_results = recipes_collection.query(
        query_texts=[query_text],
        n_results=20,
        include=['documents', 'metadatas']
    )
    reranked = rerank(
        query_text,
        recipe_results['documents'][0],
        recipe_results['metadatas'][0],
        top_k=20
    )
    
    recipe_list = []
    for idx, (doc, meta, score) in enumerate(reranked):
        recipe_list.append({
            "index": idx,
            "name": meta['name'],
            "document": doc,
            "metadata": meta,
            "url": meta.get('url', 'N/A'),
            "score": score
        })
    
    # Apply boosting
    recipe_list = boost_recipes(query_text, recipe_list)
    # Keep only boosted recipes if any got a boost
    if any(r.get("boost", 0) > 0 for r in recipe_list):
        recipe_list = [r for r in recipe_list if r.get("boost", 0) > 0]
    # Sort by (boost, score)
    recipe_list.sort(key=lambda x: (x.get("boost", 0), x["score"]), reverse=True)
    
    return recipe_list[:n_results]

def process_selected_recipe(query_text, selected_recipe):
    recipe_doc = selected_recipe["document"]
    recipe_meta = selected_recipe["metadata"]
    
    nutritional_data = "Not Available"
    if "Nutritional Info" in recipe_doc:
        nutritional_data = (
            recipe_doc.split("Nutritional Info:")[-1]
            .strip()
            .split("\n\n")[0]
            .strip()
        )

    ingredients_keywords = extract_ingredients(recipe_doc)
    ingredients_from_db = {
        ing: search_ingredients_chroma(ing, desired=3)
        for ing in ingredients_keywords
    }

    prompt = generate_prompt(
        user_query=query_text,
        recipe_name=recipe_meta['name'],
        recipe_url=recipe_meta.get('url', 'N/A'),
        recipe_details=recipe_doc,
        nutritional_data=nutritional_data,
        ingredients_from_db=ingredients_from_db
    )

    llm_response = get_llm_response(prompt)
    return {
        "question": query_text,
        "answer": llm_response,
        "contexts": [
            f"Recipe Details: {recipe_doc}",
            f"Nutritional Information: {nutritional_data}"
        ] + [
            (
                f"FairPrice Ingredient: {prod['metadata']['name']} "
                f"by {prod['metadata']['brand']} "
                f"(Price: ${prod['metadata']['price']}, "
                f"Size: {prod['metadata']['size']}, "
                f"URL: {prod['metadata'].get('url', 'N/A')})"
            )
            for ing, prods in ingredients_from_db.items()
            for prod in prods
            if prod['metadata'].get('url', 'N/A') == 'N/A'
               or is_valid_url(prod['metadata'].get('url', 'N/A'))
        ]
    }

def query_all(query_text):
    # For demonstration or direct calls
    recipe_choices = get_recipe_choices(query_text, n_results=3)
    if not recipe_choices:
        return "No recipes found for your query."
    selected = recipe_choices[0]
    result = process_selected_recipe(query_text, selected)
    return result["answer"]


if __name__ == "__main__":
    # Simple test
    test_queries = [
        "high protein tofu dish",
        "low carb vegetarian meal",
        "halal tom yam soup under $3",
        "quick chicken stir fry",
        "dairy free breakfast ideas",
        "cheap vegan lunch",
        "keto friendly snacks",
        "gluten free dinner",
        "low sodium soup",
        "iron rich meals for vegetarians"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        ans = query_all(q)
        print(ans)
