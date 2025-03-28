import sqlite3
import pandas as pd
import sys

def reset_ids(database_path, table_name):
    """Resets the ID column to be 1, 2, 3... in the given SQLite database table."""
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Drop existing ID column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Insert new ID column
    df.insert(0, 'id', range(1, len(df) + 1))

    # Overwrite table
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    conn.close()

    print(f"✅ ID reset for '{table_name}' in '{database_path}' ({len(df)} rows)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ResetID.py <database_path> <table_name>")
    else:
        db_path = sys.argv[1]
        table_name = sys.argv[2]
        reset_ids(db_path, table_name)

#how to run
#python ResetID.py recipes_clean.db recipes_clean
#python ResetID.py recipes_final.db recipes

