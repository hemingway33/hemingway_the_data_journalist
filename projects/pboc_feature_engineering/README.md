# PBOC Data SQL Database and Feature Engineering

This directory contains scripts to create a SQLite database with all the People's Bank of China (PBOC) credit report data tables and extract risk features from this data.

## Files

- `pboc_tables_sql/`: Contains all SQL files with table definitions and sample data
- `create_pboc_database.py`: Script to create and populate the SQLite database
- `query_pboc_database.py`: Script to query and explore the database
- `feature_factory.py`: Abstract factory class for implementing credit risk features
- `custom_features.py`: Example implementation of custom risk features
- `pboc_data.db`: The resulting SQLite database (created when running the script)

## Usage

### Creating the Database

To create the database, run:

```bash
python create_pboc_database.py
```

This will:
1. Create a new SQLite database file called `pboc_data.db`
2. Process all SQL files in the `pboc_tables_sql` directory
3. Create tables and insert sample data
4. Log progress and any errors encountered

### Querying the Database

To explore the database, use the query script:

```bash
# List all tables
python query_pboc_database.py --list

# Describe a specific table
python query_pboc_database.py --describe pcr_marriage_info

# Execute a custom SQL query
python query_pboc_database.py --query "SELECT * FROM pcr_marriage_info WHERE marital_status_code='已婚'"
```

### Extracting Features

The project includes a feature factory framework for extracting risk features from the PBOC data:

```bash
# Run the example implementation
python custom_features.py
```

#### Feature Factory Framework

`feature_factory.py` contains:
- `PBOCFeatureFactory`: Abstract base class with core functionality
- `BasicPBOCFeatures`: Implementation of basic credit risk features
- `AdvancedPBOCFeatures`: Implementation of more sophisticated risk features

#### Custom Feature Implementation

To create your own features:

1. Create a new class inheriting from one of the base classes
2. Implement methods with names like `extract_feature_name`
3. Use the database querying framework provided by the parent class

Example:

```python
from feature_factory import AdvancedPBOCFeatures

class MyCustomFeatures(AdvancedPBOCFeatures):
    def extract_my_custom_feature(self, report_id: str) -> float:
        # Use self.query() to execute SQL queries
        # Use self.get_feature() to reuse other features
        # Return the calculated feature value
        return calculated_value
```

## Database Schema

The database contains multiple tables that store various aspects of credit reports, including:
- Personal identity information
- Marriage information
- Credit card summaries
- Credit accounts
- Career information
- And more

Each table corresponds to a specific SQL file in the `pboc_tables_sql` directory. 