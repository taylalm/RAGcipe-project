import sqlite3

# Connect to your database
conn = sqlite3.connect("recipes.db")
cursor = conn.cursor()

# Execute the reset queries
cursor.executescript("""
    CREATE TABLE temp_recipes AS SELECT name, ingredients, method, nutritional_data, url FROM recipes;
    DROP TABLE recipes;
    CREATE TABLE recipes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        ingredients TEXT,
        method TEXT,
        nutritional_data TEXT,
        url TEXT
    );
    INSERT INTO recipes (name, ingredients, method, nutritional_data, url)
    SELECT name, ingredients, method, nutritional_data, url FROM temp_recipes;
    DROP TABLE temp_recipes;
""")

# Commit changes and close connection
conn.commit()
conn.close()

print("ID column reset successfully!")
