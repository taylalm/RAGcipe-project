from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import fitz  # PyMuPDF
import os
from tqdm import tqdm
import json
import re
import sqlite3

# --- Constants ---
BASE_URL = "https://www.healthhub.sg/programmes/nutrition-hub/healthy-recipes"
PDF_DIR = "downloaded_pdfs"  
DB_PATH = "recipes.db"

# --- Step 1: Scrape Recipe PDF Links ---
def scrape_pdf_links():
    """Scrapes HealthHub website and extracts only PDF recipe links."""
    driver = webdriver.Chrome()
    driver.get(BASE_URL)
    time.sleep(3)  # Allow page to load

    pdf_links = []  

    for page in range(48):  # Iterate through all pages should be 47 test 3 first
        print(f"📌 Scraping page {page + 1}...")

        # Find "View Recipe" buttons
        recipe_buttons = driver.find_elements(By.XPATH, "//a[contains(text(),'View Recipe') and contains(@class, 'btn-rounded red f5')]")

        for recipe in recipe_buttons:
            try:
                pdf_url = recipe.get_attribute("href")  
                pdf_links.append(pdf_url)
                print(f"✅ Extracted PDF: {pdf_url}")
            except Exception as e:
                print(f"⚠️ Error extracting PDF: {e}")

        # Scroll and navigate pages
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 200);")
        time.sleep(2)
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//a[@id='navc_recipes_next-page']"))
            )
            driver.execute_script("arguments[0].click();", next_button)  
            time.sleep(3)  
        except:
            print(f"⚠️ No 'Next Page' button found or not clickable.")
            break  

    driver.quit()

    # Save links
    with open("pdf_links.txt", "w") as f:
        for link in pdf_links:
            f.write(link + "\n")

    print(f"\n🎉 Scraping complete! Extracted {len(pdf_links)} recipe PDFs.")

    # Filter links
    filtered_links = [link.strip() for link in pdf_links if "ch-api.healthhub.sg/api/public/content/" in link]
    with open("filtered_pdf_links.txt", "w") as f:
        f.write("\n".join(filtered_links))

    print(f"✅ Filtered {len(filtered_links)} PDF links and saved.")
    return filtered_links


# --- Step 2: Download PDFs & Extract Text ---
def download_and_extract_text(pdf_links):
    """Downloads PDFs and extracts text."""
    os.makedirs(PDF_DIR, exist_ok=True)
    extracted_texts = []

    for i, pdf_url in enumerate(tqdm(pdf_links, desc="Processing PDFs")):
        try:
            pdf_path = os.path.join(PDF_DIR, f"recipe_{i}.pdf")
            response = requests.get(pdf_url)
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"

            extracted_texts.append({"url": pdf_url, "text": text})
            print(f"✅ Extracted text from: {pdf_url}")

        except Exception as e:
            print(f"❌ Failed to process {pdf_url}: {e}")

    with open("extracted_recipes.json", "w", encoding="utf-8") as f:
        json.dump(extracted_texts, f, ensure_ascii=False, indent=4)

    print(f"✅ Extracted text saved to extracted_recipes.json")
    return extracted_texts


# --- Step 3: Extract Structured Recipe Data ---
import re

def extract_recipe_data(raw_text):
    """Extracts structured recipe information from text, with debugging."""

    print("\n🔍 Debugging extract_recipe_data()")  
    print(f"📌 Raw Text (First 300 chars): {raw_text[:300]}")  

    # 🔹 **Step 1: Remove "Healthier Choice Symbol" (Handles Multi-line Cases & Extra Spaces)**
    raw_text = re.sub(r"\*?\s*Choose\s+products\s+with\s+the\s+Healthier\s+Choice\s+Symbol\.*\s*", "", raw_text, flags=re.IGNORECASE)

    # 🔹 **Step 2: Extract Name (Last valid text line from the back)**
    title_candidates = re.findall(r"\n([A-Za-z\s&-]+)\n*", raw_text)

    # Remove section headers (Ensure name is not "Serves", "Ingredients", etc.)
    title_candidates = [t.strip() for t in title_candidates if not re.search(r"(?i)\b(serves|ingredients|method|tips|nutrition|prep time|cook time)\b", t)]

    # Select the last meaningful line as the name
    name = title_candidates[-1] if title_candidates else "Unknown Recipe"

    # 🔹 **Fix: Handle Newlines in Names (Join Multi-line Names)**
    name = re.sub(r"\s*\n\s*", " ", name).strip()

    print(f"✅ Extracted Name: {name}")

    # 🔹 **Step 3: Extract Ingredients**
    ingredients_match = re.search(r"(?i)ingredients\s*(.*?)\s*(?=method|tips|nutrition|$)", raw_text, re.DOTALL)
    ingredients = ingredients_match.group(1).strip() if ingredients_match else "Not Found"

    # 🔹 **Step 4: Extract Method**
    method_match = re.search(r"(?i)method\s*(.*?)\s*(?=nutrition|tips|$)", raw_text, re.DOTALL)
    method = method_match.group(1).strip() if method_match else "Not Found"

    # 🔹 **Step 5: Extract Nutrition Information (Fix Variations & Extra Spaces)**
    nutrition_match = re.search(r"(?i)(?:nutrition information|nutritional information)\s*\(?(per serving)?\)?\s*:? (.*?$)", raw_text, re.DOTALL)
    nutrition = nutrition_match.group(2).strip() if nutrition_match else "Not Available"

    # 🔹 **Standardize Nutrition Header Removal**  
    # Removes all variations of "Nutritional Information (Per Serving):"
    nutrition = re.sub(r"(?i)(nutrition information|nutritional information)\s*\(?(per serving)?\)?[:\s]*", "", nutrition).strip()

    # 🔹 **Final Cleaning for Ingredients & Method**
    def clean_text(text):
        text = re.sub(r"•\s?", "", text).strip()  # Remove bullets
        text = re.sub(r"^- ", "", text)  # Remove leading dash
        text = re.sub(r"\s*-\s*", ", ", text)  # Replace dashes with commas
        return text.strip()

    ingredients = clean_text(ingredients)
    method = clean_text(method)
    nutrition = clean_text(nutrition)

    # Debugging Prints
    print(f"✅ Extracted Ingredients: {ingredients[:200]}")  
    print(f"✅ Extracted Method: {method[:200]}")  
    print(f"✅ Extracted Nutrition: {nutrition[:200]}")  

    return {
        "name": name,
        "ingredients": ingredients,
        "method": method,
        "nutritional_data": nutrition
    }





def extract_and_structure_recipes(extracted_texts):
    """Processes raw text and extracts structured recipe data."""

    structured_recipes = []

    for recipe in extracted_texts:
        structured_data = extract_recipe_data(recipe["text"])  # Extract fields from raw text
        structured_data["url"] = recipe["url"]  # Keep the URL for database storage
        structured_recipes.append(structured_data)

    return structured_recipes



# --- Step 4: Store Data in SQLite ---
def save_to_db(structured_recipes):
    """Stores structured recipes in SQLite DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ingredients TEXT DEFAULT NULL,
            method TEXT DEFAULT NULL,
            nutritional_data TEXT DEFAULT NULL,
            url TEXT UNIQUE NOT NULL
        )
    """)

    # Insert or update records
    for recipe in structured_recipes:
        cursor.execute("""
            INSERT INTO recipes (name, ingredients, method, nutritional_data, url) 
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                name = excluded.name,
                ingredients = excluded.ingredients,
                method = excluded.method,
                nutritional_data = excluded.nutritional_data
        """, (recipe["name"], recipe["ingredients"], recipe["method"], recipe["nutritional_data"], recipe["url"]))

    conn.commit()
    conn.close()
    print("✅ Recipes stored in SQL!")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("📌 Starting HealthHub recipe scraping...")

        # Step 1: Scrape PDF Links
        pdf_links = scrape_pdf_links()
        if not pdf_links:
            print("⚠️ No PDF links found. Exiting.")
            exit()

        # Step 2: Download & Extract Text
        extracted_texts = download_and_extract_text(pdf_links)
        if not extracted_texts:
            print("⚠️ No text extracted from PDFs. Exiting.")
            exit()

        # Step 3: Extract Structured Data
        structured_recipes = extract_and_structure_recipes(extracted_texts)
        if not structured_recipes:
            print("⚠️ No structured recipes found. Exiting.")
            exit()

        # Step 4: Store in Database
        save_to_db(structured_recipes)

        print("\n🎉 Scraping and database insertion completed successfully!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
