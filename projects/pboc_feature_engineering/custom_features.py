from feature_factory import AdvancedPBOCFeatures
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class CustomPBOCFeatures(AdvancedPBOCFeatures):
    """
    Custom PBOC features implementation with additional risk indicators.
    Inherits from AdvancedPBOCFeatures and adds domain-specific features.
    """
    
    def extract_recent_query_organizations(self, report_id: str) -> List[str]:
        """
        Get list of organizations that queried this report in the last 3 months
        
        Args:
            report_id: The credit report ID
            
        Returns:
            List of organization names
        """
        query = f"""
        SELECT organization_name 
        FROM pcr_query_record_list 
        WHERE report_id = '{report_id}' 
        AND query_date >= date('now', '-3 month')
        ORDER BY query_date DESC
        """
        df = self.query(query)
        if df.empty:
            return []
        return df['organization_name'].unique().tolist()
    
    def extract_loan_approval_likelihood(self, report_id: str) -> float:
        """
        Custom feature: Estimate likelihood of loan approval based on multiple factors
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Score between 0-1 representing approval likelihood
        """
        # Get key risk factors
        credit_score = self.get_feature(report_id, "credit_score")
        has_overdue = self.get_feature(report_id, "overdue_status")
        utilization = self.get_feature(report_id, "credit_utilization")
        query_count = self.get_feature(report_id, "query_count_6m")
        
        # Simple scoring model (can be replaced with a more sophisticated one)
        base_score = min(100, credit_score) / 100  # Normalize to 0-1
        
        # Penalties
        if has_overdue:
            base_score *= 0.5  # 50% reduction if any overdue accounts
        
        if utilization > 0.7:
            base_score *= (1 - (utilization - 0.7) * 2)  # Reduce score for high utilization
            
        if query_count > 3:
            base_score *= (1 - min(0.5, (query_count - 3) * 0.1))  # Penalty for many inquiries
        
        # Ensure score is between 0 and 1
        return max(0, min(1, base_score))
    
    def extract_high_risk_flags(self, report_id: str) -> Dict[str, bool]:
        """
        Identify high-risk flags in the credit report
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Dictionary of risk flags
        """
        flags = {}
        
        # Check for bankruptcies
        bankruptcy_query = f"""
        SELECT COUNT(*) as bankruptcy_count 
        FROM pcr_credit_acc_special 
        WHERE report_id = '{report_id}' AND special_type LIKE '%破产%'
        """
        bankruptcy_df = self.query(bankruptcy_query)
        flags['bankruptcy'] = bankruptcy_df['bankruptcy_count'].iloc[0] > 0
        
        # Check for legal judgments
        judgment_query = f"""
        SELECT COUNT(*) as judgment_count 
        FROM pcr_credit_acc_special 
        WHERE report_id = '{report_id}' AND special_type LIKE '%判决%'
        """
        judgment_df = self.query(judgment_query)
        flags['legal_judgment'] = judgment_df['judgment_count'].iloc[0] > 0
        
        # Check for fraud alerts
        fraud_query = f"""
        SELECT COUNT(*) as fraud_count 
        FROM pcr_credit_tips_list 
        WHERE report_id = '{report_id}' AND content LIKE '%欺诈%'
        """
        fraud_df = self.query(fraud_query)
        flags['fraud_alert'] = fraud_df['fraud_count'].iloc[0] > 0
        
        # Check for multiple recent applications
        apps_query = f"""
        SELECT COUNT(*) as recent_apps 
        FROM pcr_query_record_list 
        WHERE report_id = '{report_id}' 
        AND query_date >= date('now', '-30 day')
        AND query_reason LIKE '%贷款%'
        """
        apps_df = self.query(apps_query)
        flags['multiple_applications'] = apps_df['recent_apps'].iloc[0] >= 3
        
        return flags
    
    def extract_credit_trend(self, report_id: str) -> str:
        """
        Analyze credit history trend
        
        Args:
            report_id: The credit report ID
            
        Returns:
            Trend assessment: 'improving', 'stable', 'deteriorating'
        """
        # Get 5-year history
        query = f"""
        SELECT 
            year, 
            month, 
            overdue_amount
        FROM pcr_credit_acc_5years 
        WHERE report_id = '{report_id}'
        ORDER BY year DESC, month DESC
        """
        df = self.query(query)
        
        if df.empty:
            return "insufficient_data"
        
        # Create a time series of overdue amounts
        df['period'] = df['year'] * 100 + df['month']  # Convert to YYYYMM format
        df = df.sort_values('period', ascending=True)
        
        # Analyze trend using simple regression
        if len(df) >= 6:
            recent = df.tail(6)
            x = np.arange(len(recent))
            y = recent['overdue_amount'].values
            
            if np.all(y == 0):
                return "stable"
                
            # Calculate slope of trend line
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 10:  # Significant increase in overdue amounts
                return "deteriorating"
            elif slope < -10:  # Significant decrease in overdue amounts
                return "improving"
            else:
                return "stable"
        
        return "insufficient_data"


# Example usage
if __name__ == "__main__":
    # Initialize feature extractor
    feature_extractor = CustomPBOCFeatures()
    
    # Get a sample report ID
    sample_report_id = feature_extractor.query("SELECT report_id FROM pcr_identity LIMIT 1")
    if not sample_report_id.empty:
        report_id = sample_report_id['report_id'].iloc[0]
        
        # Extract all features for this report ID
        features = feature_extractor.extract_all_features(report_id)
        
        # Print features
        print("\nExtracted Features:")
        print("-" * 50)
        for name, value in features.items():
            print(f"{name}: {value}")
        
        # Generate a comprehensive credit risk assessment
        print("\nCredit Risk Assessment:")
        print("-" * 50)
        
        approval_likelihood = features.get('loan_approval_likelihood', 0)
        risk_flags = features.get('high_risk_flags', {})
        credit_trend = features.get('credit_trend', 'unknown')
        
        print(f"Loan Approval Likelihood: {approval_likelihood:.2%}")
        print(f"Credit Trend: {credit_trend}")
        
        if any(risk_flags.values()):
            print("Risk Flags Detected:")
            for flag, is_active in risk_flags.items():
                if is_active:
                    print(f"  - {flag}")
        else:
            print("No major risk flags detected")
    
    # Close connection
    feature_extractor.disconnect() 