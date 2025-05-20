import os
import sqlite3
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to SQL files directory
SQL_DIR = Path(__file__).parent / "pboc_tables_sql"
# Path to create SQLite database
DB_PATH = Path(__file__).parent / "pboc_data.db"

def create_database():
    """Creates a new SQLite database and executes all SQL files"""
    # Connect to SQLite database (will be created if it doesn't exist)
    logger.info(f"Creating/connecting to database at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Process each SQL file
    sql_files = sorted([f for f in os.listdir(SQL_DIR) if f.endswith('.sql')])
    
    for sql_file in sql_files:
        file_path = SQL_DIR / sql_file
        logger.info(f"Processing {sql_file}...")
        
        try:
            # Read the SQL file content
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Extract CREATE TABLE statement
            create_pattern = r'create table\s+(\w+\.)?(\w+)'
            match = re.search(create_pattern, sql_content, re.IGNORECASE)
            if match:
                table_name = match.group(2)
                
                # Extract columns from the CREATE TABLE statement
                create_table_pattern = r'create table\s+(?:\w+\.)?(\w+)\s*\((.*?)\)(?:\s*comment\s*.*?)?(?:\s*engine\s*=\s*\w+)?;'
                table_match = re.search(create_table_pattern, sql_content, re.IGNORECASE | re.DOTALL)
                
                if table_match:
                    columns_text = table_match.group(2)
                    
                    # Clean up column definitions by removing MySQL-specific comments
                    columns_cleaned = re.sub(r'comment\s*\'.*?\'', '', columns_text)
                    # Remove any trailing commas from the last column
                    columns_cleaned = re.sub(r',\s*$', '', columns_cleaned.strip())
                    
                    # Create a clean SQLite-compatible CREATE TABLE statement
                    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_cleaned});"
                    
                    # Execute the modified CREATE TABLE statement
                    cursor.execute(create_table_sql)
                    conn.commit()
                    
                    # Extract and process INSERT statements
                    insert_statements = re.findall(r'INSERT INTO\s+\w+\.\w+\s+\(.*?\)\s+VALUES\s+\(.*?\);', sql_content, re.DOTALL | re.IGNORECASE)
                    
                    if insert_statements:
                        for stmt in insert_statements:
                            # Remove database prefix from INSERT statements
                            modified_insert = re.sub(r'INSERT INTO\s+\w+\.', f'INSERT INTO ', stmt, flags=re.IGNORECASE)
                            
                            try:
                                cursor.execute(modified_insert)
                            except sqlite3.Error as e:
                                logger.error(f"Error executing INSERT: {str(e)}")
                                logger.error(f"Statement: {modified_insert[:200]}...")
                        
                        conn.commit()
                    
                    # Verify data was inserted
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    logger.info(f"Created table {table_name} with {count} rows")
                else:
                    logger.warning(f"Could not parse CREATE TABLE statement in {sql_file}")
            else:
                logger.warning(f"Could not find CREATE TABLE statement in {sql_file}")
                
        except Exception as e:
            logger.error(f"Error processing {sql_file}: {str(e)}")
            conn.rollback()
    
    # Close the connection
    conn.close()
    logger.info("Database creation completed")

if __name__ == "__main__":
    # Delete existing database if it exists
    if DB_PATH.exists():
        logger.info(f"Removing existing database at {DB_PATH}")
        DB_PATH.unlink()
    
    create_database() 