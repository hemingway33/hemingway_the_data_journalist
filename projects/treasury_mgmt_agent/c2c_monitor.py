import pandas as pd
from datetime import datetime, timedelta

class C2CMonitor:
    """
    Monitors Cash Conversion Cycle (C2C) and related operational ratios (DSO, DIO, DPO)
    in real-time and provides alerting capabilities.
    """

    def __init__(self, company_name: str):
        """
        Initializes the C2CMonitor.

        Args:
            company_name (str): The name of the company.
        """
        self.company_name = company_name
        self.current_data = {} # To store current financial figures
        self.ratios = {}       # To store calculated ratios
        self.thresholds = {    # Default thresholds, can be customized
            'DSO': {'high': 45, 'critical': 60},
            'DIO': {'high': 60, 'critical': 90},
            'DPO': {'low': 30, 'critical_low': 20}, # Low DPO can also be a concern
            'C2C': {'high': 75, 'critical': 100}
        }
        self.alerts = []

        print(f"C2C Monitor initialized for {self.company_name}.")
        self._load_initial_data() # Load some dummy initial data

    def _load_initial_data(self):
        """
        Placeholder: Simulates loading initial financial data required for C2C calculations.
        In a real application, this would connect to ERP, accounting systems, etc.
        All figures are assumed to be for a specific period (e.g., last 365 days or YTD).
        """
        print("Loading initial financial data (e.g., from ERP/Accounting System)...")
        # These are annual figures for simplicity in this example.
        # For real-time, you'd likely use average balances over a shorter period (e.g., 90 days)
        # and corresponding revenue/COGS for that period.
        self.current_data = {
            'period_days': 365, # Assuming annual data for these initial figures
            'revenue': 10000000, # Total credit sales for the period
            'cogs': 6000000,    # Cost of Goods Sold for the period
            'average_accounts_receivable': 1200000,
            'average_inventory': 1500000,
            'average_accounts_payable': 800000,
            'last_updated': datetime.now()
        }
        print(f"Initial data loaded: {self.current_data}")
        self.calculate_all_ratios()

    def update_financial_data(self, new_data: dict):
        """
        Updates the financial data used for C2C calculations.

        Args:
            new_data (dict): A dictionary containing the latest financial figures.
                             Expected keys: 'revenue', 'cogs', 'average_accounts_receivable',
                                            'average_inventory', 'average_accounts_payable'.
                             It should also ideally include 'period_days' if the period changes.
        """
        print(f"Updating financial data at {datetime.now()}...")
        required_keys = ['revenue', 'cogs', 'average_accounts_receivable', 'average_inventory', 'average_accounts_payable']
        for key in required_keys:
            if key not in new_data:
                print(f"Warning: '{key}' not found in new_data. Using previous value if available.")
                if key not in self.current_data: # If no previous value either
                     print(f"Critical Error: '{key}' is missing and no previous value exists. Cannot update.")
                     return False

        self.current_data.update(new_data)
        self.current_data['last_updated'] = datetime.now()
        print(f"Data updated: {self.current_data}")
        self.calculate_all_ratios()
        self.check_thresholds_and_alert()
        return True

    def _calculate_dso(self) -> float:
        """Calculates Days Sales Outstanding (DSO)."""
        if self.current_data.get('revenue', 0) == 0:
            return float('inf') # Avoid division by zero
        period_days = self.current_data.get('period_days', 365)
        return (self.current_data.get('average_accounts_receivable', 0) / self.current_data['revenue']) * period_days

    def _calculate_dio(self) -> float:
        """Calculates Days Inventory Outstanding (DIO)."""
        if self.current_data.get('cogs', 0) == 0:
            return float('inf') # Avoid division by zero
        period_days = self.current_data.get('period_days', 365)
        return (self.current_data.get('average_inventory', 0) / self.current_data['cogs']) * period_days

    def _calculate_dpo(self) -> float:
        """Calculates Days Payable Outstanding (DPO)."""
        if self.current_data.get('cogs', 0) == 0:
            return float('inf') # Avoid division by zero
        period_days = self.current_data.get('period_days', 365)
        return (self.current_data.get('average_accounts_payable', 0) / self.current_data['cogs']) * period_days

    def _calculate_c2c(self) -> float:
        """Calculates Cash Conversion Cycle (C2C)."""
        dso = self.ratios.get('DSO', 0)
        dio = self.ratios.get('DIO', 0)
        dpo = self.ratios.get('DPO', 0)
        if any(val == float('inf') for val in [dso, dio, dpo]):
            return float('inf')
        return dso + dio - dpo

    def calculate_all_ratios(self):
        """Calculates all C2C related ratios and stores them."""
        self.ratios['DSO'] = round(self._calculate_dso(), 2)
        self.ratios['DIO'] = round(self._calculate_dio(), 2)
        self.ratios['DPO'] = round(self._calculate_dpo(), 2)
        self.ratios['C2C'] = round(self._calculate_c2c(), 2)
        print(f"Ratios calculated: {self.ratios}")

    def set_thresholds(self, ratio_name: str, high_threshold: float = None, critical_threshold: float = None, low_threshold: float = None, critical_low_threshold: float = None):
        """
        Sets monitoring thresholds for a specific ratio.
        For DPO, low thresholds indicate potential issues (paying too quickly).
        """
        if ratio_name not in self.thresholds:
            self.thresholds[ratio_name] = {}

        if high_threshold is not None:
            self.thresholds[ratio_name]['high'] = high_threshold
        if critical_threshold is not None:
            self.thresholds[ratio_name]['critical'] = critical_threshold
        if low_threshold is not None: # Specifically for DPO or similar metrics
            self.thresholds[ratio_name]['low'] = low_threshold
        if critical_low_threshold is not None:
            self.thresholds[ratio_name]['critical_low'] = critical_low_threshold

        print(f"Thresholds for {ratio_name} updated to: {self.thresholds[ratio_name]}")


    def check_thresholds_and_alert(self):
        """Checks current ratios against thresholds and generates alerts."""
        self.alerts = [] # Clear previous alerts for this check
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for ratio_name, value in self.ratios.items():
            if value == float('inf'):
                alert_msg = f"ALERT [{timestamp}]: {ratio_name} is exceptionally high (possibly due to zero denominator). Value: {value}"
                self.alerts.append(alert_msg)
                print(alert_msg)
                continue

            threshold_config = self.thresholds.get(ratio_name, {})

            # Check high thresholds (DSO, DIO, C2C)
            critical_high = threshold_config.get('critical')
            high = threshold_config.get('high')

            if critical_high is not None and value >= critical_high:
                alert_msg = f"CRITICAL ALERT [{timestamp}]: {ratio_name} is {value}, exceeding critical threshold of {critical_high}."
                self.alerts.append(alert_msg)
                print(alert_msg)
            elif high is not None and value >= high:
                alert_msg = f"ALERT [{timestamp}]: {ratio_name} is {value}, exceeding high threshold of {high}."
                self.alerts.append(alert_msg)
                print(alert_msg)

            # Check low thresholds (primarily for DPO)
            critical_low = threshold_config.get('critical_low')
            low = threshold_config.get('low')

            if critical_low is not None and value <= critical_low:
                alert_msg = f"CRITICAL ALERT [{timestamp}]: {ratio_name} is {value}, below critical low threshold of {critical_low}."
                self.alerts.append(alert_msg)
                print(alert_msg)
            elif low is not None and value <= low:
                alert_msg = f"ALERT [{timestamp}]: {ratio_name} is {value}, below low threshold of {low}."
                self.alerts.append(alert_msg)
                print(alert_msg)

        if not self.alerts:
            print(f"[{timestamp}]: All ratios within defined thresholds.")
        return self.alerts

    def get_current_ratios(self) -> dict:
        """Returns the currently calculated ratios."""
        return self.ratios

    def get_alerts(self) -> list:
        """Returns the list of current alerts."""
        return self.alerts

    def simulate_real_time_monitoring(self, num_updates: int = 5, interval_seconds: int = 1):
        """
        Simulates real-time data updates and monitoring.
        In a real system, this would be event-driven or on a scheduler.
        """
        import time
        print("\n--- Starting Real-Time C2C Monitoring Simulation ---")
        for i in range(num_updates):
            print(f"\n--- Monitoring Update {i+1}/{num_updates} ---")
            # Simulate changes in financial data
            # For simplicity, we'll slightly modify existing values
            new_data_sim = self.current_data.copy()
            new_data_sim['average_accounts_receivable'] *= (1 + (i % 2 * 0.05 - 0.02)) # Fluctuate by +/- 2-3%
            new_data_sim['average_inventory'] *= (1 + (i % 2 * 0.03 - 0.01))
            new_data_sim['average_accounts_payable'] *= (1 + (i % 2 * 0.04 - 0.025))
            new_data_sim['revenue'] *= (1.001) # Slight increase in revenue
            new_data_sim['cogs'] *= (1.0005)   # Slight increase in COGS

            self.update_financial_data(new_data_sim)
            time.sleep(interval_seconds)
        print("\n--- Real-Time C2C Monitoring Simulation Complete ---")

if __name__ == '__main__':
    monitor = C2CMonitor(company_name="SupplyChain Inc.")

    print("\n--- Initial Ratios ---")
    print(monitor.get_current_ratios())
    monitor.check_thresholds_and_alert() # Check initial thresholds

    print("\n--- Customizing Thresholds for DSO ---")
    monitor.set_thresholds(ratio_name='DSO', high_threshold=40, critical_threshold=50)
    monitor.check_thresholds_and_alert() # Re-check with new DSO thresholds

    print("\n--- Customizing Thresholds for DPO (Low values are bad) ---")
    monitor.set_thresholds(ratio_name='DPO', low_threshold=25, critical_low_threshold=15)
    monitor.check_thresholds_and_alert() # Re-check with new DPO thresholds


    # Simulate an update with significantly worse DSO
    print("\n--- Simulating Data Update: High DSO ---")
    bad_dso_data = monitor.current_data.copy() # Start from current
    bad_dso_data['average_accounts_receivable'] *= 1.5 # Increase AR significantly
    monitor.update_financial_data(bad_dso_data)

    # Simulate an update with critically low DPO
    print("\n--- Simulating Data Update: Critically Low DPO ---")
    bad_dpo_data = monitor.current_data.copy()
    bad_dpo_data['average_accounts_payable'] *= 0.3 # Decrease AP significantly
    monitor.update_financial_data(bad_dpo_data)

    # Run simulation of multiple updates
    monitor.simulate_real_time_monitoring(num_updates=3, interval_seconds=1)

    print("\n--- Final Alerts Generated During Simulations ---")
    for alert in monitor.get_alerts(): # This will only show alerts from the LAST check in simulation
        print(alert)
    # To see all alerts from simulation, you would need to collect them within the loop

    print("\n--- C2C Monitor Example Usage Complete ---")
