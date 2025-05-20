import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PBOCFeatureFactory(ABC):
    """
    Abstract base class for extracting risk features from PBOC credit report data.
    Subclasses should implement specific feature extraction methods.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the feature factory with database connection.
        
        Args:
            db_path: Path to SQLite database. If None, uses default path.
        """
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "pboc_data.db")
        else:
            self.db_path = db_path
            
        self.conn = None
        self._feature_cache = {}
        
    def connect(self):
        """Connect to the SQLite database"""
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_path)
                logger.info(f"Connected to database: {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"Database connection error: {str(e)}")
                raise
    
    def disconnect(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            
        Returns:
            DataFrame with query results
        """
        self.connect()
        try:
            return pd.read_sql(sql, self.conn)
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            logger.error(f"Query: {sql}")
            raise
    
    def get_feature(self, report_id: str, feature_name: str) -> Any:
        """
        Get a specific feature value for a report ID.
        
        Args:
            report_id: The credit report ID
            feature_name: Name of the feature to extract
            
        Returns:
            The extracted feature value
        """
        # Check if feature extraction method exists
        if not hasattr(self, f"extract_{feature_name}"):
            raise ValueError(f"Feature '{feature_name}' is not implemented")
        
        # Use cache if available
        cache_key = f"{report_id}_{feature_name}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Extract feature
        feature_method = getattr(self, f"extract_{feature_name}")
        feature_value = feature_method(report_id)
        
        # Cache the result
        self._feature_cache[cache_key] = feature_value
        return feature_value
    
    def get_features(self, report_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """
        Get multiple features for a report ID.
        
        Args:
            report_id: The credit report ID
            feature_names: List of feature names to extract
            
        Returns:
            Dictionary mapping feature names to their values
        """
        return {name: self.get_feature(report_id, name) for name in feature_names}
    
    def extract_all_features(self, report_id: str) -> Dict[str, Any]:
        """
        Extract all implemented features for a report ID.
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Dictionary mapping all feature names to their values
        """
        # Find all methods starting with 'extract_'
        feature_methods = [method[8:] for method in dir(self) 
                          if method.startswith('extract_') and callable(getattr(self, method))]
        return self.get_features(report_id, feature_methods)
    
    def batch_extract_features(self, report_ids: List[str], feature_names: List[str]) -> pd.DataFrame:
        """
        Extract features for multiple report IDs.
        
        Args:
            report_ids: List of report IDs
            feature_names: List of feature names to extract
            
        Returns:
            DataFrame with report IDs as index and features as columns
        """
        results = []
        for report_id in report_ids:
            features = self.get_features(report_id, feature_names)
            features['report_id'] = report_id
            results.append(features)
        
        df = pd.DataFrame(results)
        return df.set_index('report_id')
    
    @abstractmethod
    def extract_credit_score(self, report_id: str) -> int:
        """Extract credit score from report"""
        pass
    
    @abstractmethod
    def extract_overdue_status(self, report_id: str) -> bool:
        """Extract if customer has any overdue accounts"""
        pass


class BasicPBOCFeatures(PBOCFeatureFactory):
    """
    Implementation of basic PBOC credit report features.
    """
    
    def extract_credit_score(self, report_id: str) -> int:
        """
        Extract credit score from report
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Credit score as integer
        """
        query = f"""
        SELECT score 
        FROM pcr_score_info 
        WHERE report_id = '{report_id}'
        """
        df = self.query(query)
        if df.empty:
            return 0
        return int(df['score'].iloc[0]) if not pd.isna(df['score'].iloc[0]) else 0
    
    def extract_overdue_status(self, report_id: str) -> bool:
        """
        Check if customer has any overdue accounts
        
        Args:
            report_id: The credit report ID
            
        Returns:
            True if has overdue accounts, False otherwise
        """
        query = f"""
        SELECT COUNT(*) as overdue_count 
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}' AND overdue_amount > 0
        """
        df = self.query(query)
        return df['overdue_count'].iloc[0] > 0
    
    def extract_num_credit_accounts(self, report_id: str) -> int:
        """
        Count number of credit accounts
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Number of credit accounts
        """
        query = f"""
        SELECT COUNT(*) as account_count 
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}'
        """
        df = self.query(query)
        return int(df['account_count'].iloc[0])
    
    def extract_total_credit_limit(self, report_id: str) -> float:
        """
        Calculate total credit limit across all accounts
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Total credit limit
        """
        query = f"""
        SELECT SUM(credit_limit) as total_limit 
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}'
        """
        df = self.query(query)
        return float(df['total_limit'].iloc[0]) if not pd.isna(df['total_limit'].iloc[0]) else 0.0
    
    def extract_credit_utilization(self, report_id: str) -> float:
        """
        Calculate credit utilization ratio
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Credit utilization ratio (0-1)
        """
        query = f"""
        SELECT 
            SUM(balance) as total_balance,
            SUM(credit_limit) as total_limit
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}' AND credit_limit > 0
        """
        df = self.query(query)
        if df.empty or df['total_limit'].iloc[0] == 0:
            return 0.0
        
        balance = float(df['total_balance'].iloc[0]) if not pd.isna(df['total_balance'].iloc[0]) else 0.0
        limit = float(df['total_limit'].iloc[0]) if not pd.isna(df['total_limit'].iloc[0]) else 0.0
        
        if limit == 0:
            return 0.0
        
        utilization = balance / limit
        return min(1.0, max(0.0, utilization))  # Ensure it's between 0 and 1
    
    def extract_has_mortgage(self, report_id: str) -> bool:
        """
        Check if customer has mortgage loans
        
        Args:
            report_id: The credit report ID
            
        Returns:
            True if has mortgage, False otherwise
        """
        query = f"""
        SELECT COUNT(*) as mortgage_count 
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}' AND account_type = '房贷'
        """
        df = self.query(query)
        return df['mortgage_count'].iloc[0] > 0
    
    def extract_query_count_6m(self, report_id: str) -> int:
        """
        Count number of credit queries in last 6 months
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Number of queries
        """
        query = f"""
        SELECT COUNT(*) as query_count 
        FROM pcr_query_record_list 
        WHERE report_id = '{report_id}' 
        AND query_date >= date('now', '-6 month')
        """
        df = self.query(query)
        return int(df['query_count'].iloc[0])


class AdvancedPBOCFeatures(BasicPBOCFeatures):
    """
    Implementation of advanced PBOC credit report features.
    Inherits basic features and adds more complex ones.
    """
    
    def extract_worst_overdue_status(self, report_id: str) -> int:
        """
        Get worst overdue status (max overdue months)
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Maximum number of months overdue
        """
        query = f"""
        SELECT MAX(overdue_months) as max_overdue 
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}'
        """
        df = self.query(query)
        return int(df['max_overdue'].iloc[0]) if not pd.isna(df['max_overdue'].iloc[0]) else 0
    
    def extract_account_age(self, report_id: str) -> float:
        """
        Calculate average age of accounts in years
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Average account age in years
        """
        query = f"""
        SELECT 
            open_date,
            date('now') as current_date
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}'
        """
        df = self.query(query)
        if df.empty:
            return 0.0
        
        df['open_date'] = pd.to_datetime(df['open_date'])
        df['current_date'] = pd.to_datetime(df['current_date'])
        df['age_days'] = (df['current_date'] - df['open_date']).dt.days
        
        avg_age_days = df['age_days'].mean()
        return avg_age_days / 365.25  # Convert to years
    
    def extract_debt_to_income(self, report_id: str) -> float:
        """
        Calculate debt to income ratio
        Assumes income data is available in applicant table
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Debt to income ratio
        """
        # Get total debt
        debt_query = f"""
        SELECT SUM(balance) as total_debt 
        FROM pcr_credit_acc 
        WHERE report_id = '{report_id}'
        """
        debt_df = self.query(debt_query)
        total_debt = float(debt_df['total_debt'].iloc[0]) if not pd.isna(debt_df['total_debt'].iloc[0]) else 0.0
        
        # Get income (assuming it's in applicant info)
        income_query = f"""
        SELECT annual_income 
        FROM pcr_applicant_info 
        WHERE report_id = '{report_id}'
        """
        income_df = self.query(income_query)
        
        if income_df.empty or pd.isna(income_df['annual_income'].iloc[0]) or income_df['annual_income'].iloc[0] == 0:
            return 0.0
        
        annual_income = float(income_df['annual_income'].iloc[0])
        
        if annual_income == 0:
            return 0.0
            
        return total_debt / annual_income
    
    def extract_payment_history_pattern(self, report_id: str) -> str:
        """
        Create a pattern of recent payment history
        
        Args:
            report_id: The credit report ID
            
        Returns:
            String pattern like "OOOOXOXX" where O=on-time, X=late, ?=unknown
        """
        query = f"""
        SELECT 
            status_24m, status_23m, status_22m, status_21m, status_20m, status_19m
        FROM pcr_credit_acc_5years 
        WHERE report_id = '{report_id}'
        """
        df = self.query(query)
        
        if df.empty:
            return "??????"
        
        # Take the first row as a sample
        row = df.iloc[0]
        
        # Convert status codes to O/X/? pattern
        pattern = ""
        for i in range(24, 18, -1):
            col = f"status_{i}m"
            if col in row and not pd.isna(row[col]):
                status = row[col]
                if status == '正常':
                    pattern += "O"
                elif status == '逾期':
                    pattern += "X"
                else:
                    pattern += "?"
            else:
                pattern += "?"
                
        return pattern


# Example usage:
if __name__ == "__main__":
    # Initialize feature extractor
    feature_extractor = AdvancedPBOCFeatures()
    
    # Get a sample report ID
    sample_report_id = feature_extractor.query("SELECT report_id FROM pcr_identity LIMIT 1")
    if not sample_report_id.empty:
        report_id = sample_report_id['report_id'].iloc[0]
        
        # Extract all features for this report ID
        features = feature_extractor.extract_all_features(report_id)
        
        # Print features
        for name, value in features.items():
            print(f"{name}: {value}")
    
    # Close connection
    feature_extractor.disconnect()
