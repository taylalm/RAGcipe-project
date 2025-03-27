import tiktoken
import pandas as pd
import sqlite3

# Load your data
conn = sqlite3.connect("recipes.db")
df = pd.read_sql_query("SELECT * FROM recipes", conn)
conn.close()

# Combine fields
df['combined_text'] = (
    "Recipe Name: " + df['name'] + "\n"
    "Ingredients: " + df['ingredients'] + "\n"
    "Method: " + df['method'] + "\n"
    "Nutritional Info: " + df['nutritional_data']
)

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

# Count tokens per recipe
df['token_count'] = df['combined_text'].apply(lambda x: len(encoding.encode(x)))

# Display recipes with high token counts
print(df[['id', 'name', 'token_count']].sort_values(by='token_count', ascending=False).head(10))

# Optional: Save to CSV for inspection
df[['id', 'name', 'token_count']].to_csv("token_counts.csv", index=False)




import pandas as pd

# Load your CSV (replace with your actual file name)
df = pd.read_csv('token_counts.csv')

# Filter recipes with token_count > 1000
high_token_recipes = df[df['token_count'] > 1000]

# Display count
print(f"Total recipes with token_count > 1000: {len(high_token_recipes)}")

# Optionally display which recipes
print(high_token_recipes[['id', 'name', 'token_count']])

# Save to CSV for inspection if needed
high_token_recipes.to_csv('high_token_recipes.csv', index=False)

