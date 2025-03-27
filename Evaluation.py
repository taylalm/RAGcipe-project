from Full_Prompt_new import query_all
from tqdm import tqdm
import json

# Define a set of test queries (expand this list for a more robust evaluation)
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

def evaluate_queries():
    dataset = []
    for query in tqdm(test_queries, desc="Evaluating queries"):
        print(f"\n=== Evaluating Query: {query} ===")
        result = query_all(query)
        print("\n--- LLM Response ---")
        print(result["answer"])
        print("\n" + "="*80 + "\n")
        dataset.append(result)
    
    # Save the dataset to a JSON file (this dataset now contains only reference-free fields)
    with open("ragcipe_ragas_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Dataset saved to ragcipe_ragas_dataset.json")

if __name__ == "__main__":
    evaluate_queries()
