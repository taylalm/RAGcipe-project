import json
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate

# Load your pre-collected dataset from Evaluation.py
with open("ragcipe_ragas_dataset.json", "r") as f:
    data = json.load(f)

# Convert the list of dictionaries into a HuggingFace Dataset
hf_dataset = Dataset.from_list(data)

# Define the metric to evaluate (only faithfulness is reference-free here)
metrics_to_use = [faithfulness]

# Run the evaluation
try:
    results = evaluate(hf_dataset, metrics=metrics_to_use)
except Exception as e:
    print(f"Error during evaluation: {e}")
    results = None

# Print the evaluation results
print("RAGAS Evaluation Results:")
print(results)
