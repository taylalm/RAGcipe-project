import sqlite3

# Connect to the database
conn = sqlite3.connect("recipes.db")
cursor = conn.cursor()

# Delete all records
cursor.execute("DELETE FROM recipes;")

# Reset auto-incrementing ID
cursor.execute("DELETE FROM sqlite_sequence WHERE name='recipes';")

# Optimize database size
conn.commit()
conn.execute("VACUUM;")
conn.close()

print("Database cleared successfully!")
