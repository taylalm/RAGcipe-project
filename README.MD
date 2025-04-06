# 🥗 RAGCipe: An AI-Powered Cooking & Shopping Assistant

**RAGCipe** is an end-to-end Retrieval-Augmented Generation (RAG) system built to promote healthy, affordable eating habits by turning static government-backed recipe databases into an interactive, AI-powered experience. Using web scraping, semantic search, and LLMs, we ensure that the app suggests nutritious recipes and matches them with affordable FairPrice grocery products — complete with cost and nutritional breakdowns.

🔗 [Live Demo (Streamlit App)](https://ragcipe-project-fevq9dkjzg3ncciwqbwfcu.streamlit.app/)

---

## 📌 Problem Statement

Despite the availability of healthy and affordable recipes published by organizations like the **Singapore Heart Foundation (SHF)** and **HealthHub**, public adoption is low due to limited accessibility. **RAGCipe bridges this gap** by transforming curated recipes into a conversational interface that’s personalized, local, and easy to use — especially for working adults, caregivers, and health-conscious individuals in Singapore.

---

## 🎯 Project Objectives

- Boost awareness and adoption of healthy eating by surfacing expert-curated recipes.
- Match ingredients with **locally available** and affordable FairPrice products.
- Provide **nutritional analysis** and **cost estimation** to support informed decisions.
- Enable **natural language search** for recipes using advanced semantic retrieval.

---

## 🧑‍🍳 Data Collection & Preparation

### 🥘 1. Healthy Recipe Dataset

**Sources:**
- **Singapore Heart Foundation (SHF):** Scraped HTML pages using Selenium + BeautifulSoup.
- **HealthHub:** Scraped recipe PDFs using Selenium + PyMuPDF.

**Extracted Fields:**
- Recipe name
- Ingredients
- Cooking method
- Nutritional information
- Source URL

Recipes were stored in `recipes.db` (SQLite), with each entry cleaned and normalized.

### 🛒 2. FairPrice Product Dataset

Scraped 3,981 grocery items from **NTUC FairPrice** across:
- Rice, Noodles & Cooking Ingredients
- Meat & Seafood
- Fruits & Vegetables

**Scraped Fields:**
- Product name, size, price, ratings, brand, origin
- Key info, dietary labels, nutrition (JSON), ingredients
- URL

All product data stored in a second SQLite DB and embedded using OpenAI for semantic search.

---

## 🔍 Data Embedding & Processing

- Combined recipe fields into a single text block.
- Filtered to ≤1000 tokens using `tiktoken`.
- Embedded with `text-embedding-ada-002` into ChromaDB `recipes_collection`.
- Grocery products embedded into separate `fairprice_products_openai` ChromaDB.

---

## 🧠 System Architecture

### 1. **Query & Recipe Retrieval**
- Embed user query with `text-embedding-ada-002`.
- Semantic similarity search in `recipes_collection`.

### 2. **Cross-Encoder Reranking**
- Rerank top matches using `ms-marco-MiniLM-L-6-v2` CrossEncoder.
- Return top 5 recipes.

### 3. **FairPrice Product Matching**
- Extract ingredients from selected recipe.
- For each, retrieve top 3 FairPrice matches via vector search in `fairprice_products_openai`.

### 4. **LLM Generation (GPT-4o)**
Final prompt includes:
- Recipe name + steps
- Matched FairPrice products
- Nutrition info

LLM generates:
1. Recipe summary with link  
2. Affordable grocery list  
3. Nutrition analysis  
4. Per-serving cost estimate

---

## 🖥️ Streamlit Interface

Users can:
- Enter a prompt like _"tofu stir fry under 3 dollars"_.
- View 5 AI-ranked recipe suggestions.
- Select a recipe to view:
  - 🔗 Source summary  
  - 🛒 Grocery list with links & price  
  - 🧾 Nutrition summary  
  - 💸 Cost estimate

---

## 📈 Evaluation

### ✅ **Recipe Retrieval Accuracy**
- Evaluated Top-1, Top-3, Top-5 retrieval across 22 user queries.
- Achieved ~80% Top-1 accuracy.

Example:

**Prompt:** "What can I make with prawns and garlic?"  
**Top Recipe:** Soy Sauce Garlic Prawns (Score: 3.98)

### ✅ **Ingredient Matching (RAGAS)**
- Evaluated 10 prompts using `faithfulness` from RAGAS.
- Achieved **0.7607 faithfulness score**, showing strong consistency between matched grocery items and recipe context.

---

## ⚠️ Challenges

- Semantic mismatches: e.g., “tofu” vs “beancurd”.
- Token length filtering led to excluded recipes.
- Embedding ambiguity: “tomato” sometimes matched “tomato paste”.

---

## 🚀 Future Enhancements

- **Nutritional SQL agent:** Let users ask for "low calorie meals" or "iron-rich snacks".
- **User preference learning:** Personalize based on past queries or dietary goals.
- **Better scraping & chunking:** Recover lost recipes and support longer inputs.
- **Multilingual support** for wider accessibility.

---

## 🧪 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run Streamlit_App.py
