import sqlite3
import pandas as pd
from pathlib import Path
import argparse

# Path to SQLite database
DB_PATH = Path(__file__).parent / "pboc_data.db"

def list_tables():
    """List all tables in the database with row counts"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"{'Table Name':<40} {'Row Count':<10}")
    print("-" * 50)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{table_name:<40} {count:<10}")
    
    conn.close()

def describe_table(table_name):
    """Show table schema and sample data"""
    conn = sqlite3.connect(str(DB_PATH))
    
    # Get table schema
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print(f"\nSchema for table: {table_name}")
    print(f"{'Column':<30} {'Type':<15} {'Nullable':<10} {'Primary Key':<15}")
    print("-" * 70)
    
    for col in columns:
        cid, name, type_name, notnull, dflt_value, pk = col
        is_nullable = "NOT NULL" if notnull else "NULL"
        is_pk = "PK" if pk else ""
        print(f"{name:<30} {type_name:<15} {is_nullable:<10} {is_pk:<15}")
    
    # Show sample data
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(f"\nSample data (5 rows) for table: {table_name}")
        print(df)
    except Exception as e:
        print(f"Error fetching sample data: {str(e)}")
    
    conn.close()

def execute_query(query):
    """Execute a custom SQL query"""
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        df = pd.read_sql(query, conn)
        print(f"\nQuery results:")
        print(df)
    except Exception as e:
        print(f"Error executing query: {str(e)}")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query PBOC SQLite database")
    parser.add_argument("--list", action="store_true", help="List all tables")
    parser.add_argument("--describe", type=str, help="Describe a specific table")
    parser.add_argument("--query", type=str, help="Execute a custom SQL query")
    
    args = parser.parse_args()
    
    if args.list:
        list_tables()
    elif args.describe:
        describe_table(args.describe)
    elif args.query:
        execute_query(args.query)
    else:
        parser.print_help() 