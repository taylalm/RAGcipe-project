import os
from retrieval import get_recipe_choices
from enrichment import enrich_ingredients_with_fairprice

def main():
    # Example test prompt
    prompt = "High protein and low carb egg dish"
    
    # Choose mode
    mode = "user"  # Change to "eval" to skip enrichment
    top_k = 5

    print(f"🔍 Running in mode: {mode}")
    
    # Step 1️⃣: Get top recipes from RAG reranking
    top_recipes = get_recipe_choices(prompt, top_k=top_k, return_ids=True)
    print("📄 Top recipes retrieved:")
    for r in top_recipes:
        print(f"- {r['id']} | {r['name']}")

    if mode == "user":
        # Step 2️⃣: Enrich ingredients with FairPrice matches
        enriched = enrich_ingredients_with_fairprice(top_recipes)
        print("\n🛒 Enriched ingredient suggestions:")
        for item in enriched:
            print(f"\n🍽️ {item['name']}")
            for ing, matches in item['matches'].items():
                print(f"  🔸 {ing}")
                for m in matches:
                    print(f"    🛍️ {m['product']} (${m['price']})")

if __name__ == "__main__":
    main()
