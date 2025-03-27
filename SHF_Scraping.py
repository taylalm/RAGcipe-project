import sqlite3
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL
BASE_URL = "https://www.myheart.org.sg"
RECIPE_PAGE = f"{BASE_URL}/recipes-all/"

# Database Setup
DB_NAME = "recipes.db"
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    ingredients TEXT DEFAULT NULL,
    method TEXT DEFAULT NULL,
    nutritional_data TEXT DEFAULT NULL,
    url TEXT UNIQUE NOT NULL
)''')

conn.commit()


# --- Create a Selenium Driver ---
def create_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=options)

# --- Function to Scrape Ingredients ---
def extract_ingredients(soup):
    """Extracts ingredients from the recipe page, handling different formats."""
    ingredients = []

    # Find the "Ingredients" heading (supports <strong> and <h3>, and removes spaces)
    ingredients_heading = soup.find(lambda tag: tag.name in ["strong", "h3"] and tag.get_text(strip=True).lower() == "ingredients")

    if ingredients_heading:
        # Search for the next list (ul or ol) anywhere after "Ingredients"
        ingredients_list = ingredients_heading.find_next(["ul", "ol"])

        # Case 1: Ingredients are inside a <ul> or <ol>
        if ingredients_list:
            ingredients = [li.get_text(strip=True) for li in ingredients_list.find_all("li")]

        # Case 2: No list found, check for <p> tags with <br> separators
        else:
            next_paragraph = ingredients_heading.find_next("p")
            if next_paragraph:
                ingredients = [text.strip() for text in next_paragraph.stripped_strings if text.strip()]

    return "\n".join(ingredients)  # Return as a string with newline-separated items




# --- Function to Scrape Method ---
def extract_method(soup):
    """Extracts the cooking method from the recipe page, handling different formats."""
    method_steps = []

    # Find the "Method" heading (case-insensitive)
    method_heading = soup.find(lambda tag: tag.name in ["strong", "h3"] and tag.get_text(strip=True).lower() == "method")

    if method_heading:
        # Search for the next list (ul or ol) anywhere after "Method"
        method_list = method_heading.find_next(["ul", "ol"])

        # Case 1: Steps are in a <ul> or <ol>
        if method_list:
            method_steps = [li.get_text(strip=True) for li in method_list.find_all("li")]

        # Case 2: If no list found, check for <p> tags with <br> separators
        else:
            next_paragraph = method_heading.find_next("p")
            if next_paragraph:
                method_steps = [text.strip() for text in next_paragraph.stripped_strings if text.strip()]

    return "\n".join(method_steps)  # Return as a string with newline-separated items

# # --- Function to Scrape Nutrients ---
def extract_nutrients(soup):
    """Extracts nutritional data from the recipe page, prioritizing <br> separators first, then <p>/<ul>/<ol>."""
    nutrients = []

    # Find the "Nutrients Per Serving" heading
    nutrients_heading = soup.find(lambda tag: tag.name in ["strong", "h3"] and "nutrients per serving" in tag.get_text(strip=True).lower())

    if nutrients_heading:
        # Case 1: Extract text following <br> separators first
        br_texts = []
        for sibling in nutrients_heading.next_siblings:
            if sibling.name == "br":
                # Collect text nodes after <br>
                br_texts.extend([text.strip() for text in sibling.find_all_next(string=True) if text.strip()])
            elif sibling.name in ["p", "div"]:  # Stop extraction when new block appears
                break  
        if br_texts:
            nutrients = br_texts

        # Case 2: If no <br> text, check <p> after heading
        if not nutrients:
            next_element = nutrients_heading.find_next(["p", "ul", "ol"])
            if next_element:
                if next_element.name in ["ul", "ol"]:
                    nutrients = [li.get_text(strip=True) for li in next_element.find_all("li")]
                elif next_element.name == "p":
                    nutrients = [text.strip() for text in next_element.stripped_strings if text.strip()]

    return "\n".join(nutrients)  # Return as a newline-separated string





# --- Function to Scrape Recipe Details ---
def scrape_recipe(url):
    driver = create_driver()
    recipe_data = {}
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Recipe Name
        name_element = soup.find("h1")
        recipe_data["name"] = name_element.text.strip() if name_element else "Unknown"

        # Ingredients
        recipe_data["ingredients"] = extract_ingredients(soup) or "Not Available"

        # Method
        recipe_data["method"] = extract_method(soup) or "Not Available"

        # Nutritional Information
        recipe_data["nutritional_data"] = extract_nutrients(soup) or "Not Available"

        # Recipe URL
        recipe_data["url"] = url

        # Debugging: Print summary of extracted data
        print(f"Scraped: {recipe_data['name']}")
        print(f"  - Ingredients: {recipe_data['ingredients'][:50]}...")
        print(f"  - Method: {recipe_data['method'][:50]}...")
        print(f"  - Nutritional Info: {recipe_data['nutritional_data'][:50]}...")

    except Exception as e:
        print(f"Error scraping {url}: {e}")

    finally:
        driver.quit()

    return recipe_data


# --- Function to Scrape Recipe Listing Page ---
def scrape_all_recipe_links():
    driver = create_driver()
    all_recipe_links = []
    
    try:
        for page_num in range(1, 13):  # Loop through pages 1 to 12
            page_url = f"https://www.myheart.org.sg/recipes-all/page/{page_num}/"
            print(f"Loading page: {page_url}")
            driver.get(page_url)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "featuredpostbox"))
            )

            soup = BeautifulSoup(driver.page_source, "html.parser")
            recipe_elements = soup.select(".featuredpostbox a")

            print(f"Found {len(recipe_elements)} recipes on page {page_num}")

            for a in recipe_elements:
                recipe_link = urljoin(BASE_URL, a['href'])
                all_recipe_links.append(recipe_link)

    except Exception as e:
        print(f"Error scraping pages: {e}")

    finally:
        driver.quit()

    return sorted(set(all_recipe_links))  # Remove duplicates if any



# --- Main Execution ---
if __name__ == "__main__":
    print("Starting recipe scrape...")
    recipe_urls = scrape_all_recipe_links()
    print(f"Found {len(recipe_urls)} recipes.")

    scraped_recipes = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_recipe, url): url for url in recipe_urls}
        for future in as_completed(future_to_url):
            data = future.result()
            if data:
                scraped_recipes.append(data)

    # Insert into SQLite
    bulk_data = [
        (r["name"], r.get("ingredients", None), r.get("method", None), 
         r.get("nutritional_data", None), r["url"]) for r in scraped_recipes
    ]

    # Debugging: Print data before insertion
    for recipe in bulk_data:
        print(f"Saving: {recipe}")

    # Insert or update records
    for recipe in bulk_data:
        cursor.execute('''INSERT INTO recipes (name, ingredients, method, nutritional_data, url)
                          VALUES (?, ?, ?, ?, ?)
                          ON CONFLICT(url) DO UPDATE SET 
                          ingredients=excluded.ingredients,
                          method=excluded.method,
                          nutritional_data=excluded.nutritional_data''', 
                       (recipe[0], recipe[1], recipe[2], recipe[3], recipe[4]))

    conn.commit()

    print("Scraping and database insertion completed.")
    conn.close()
