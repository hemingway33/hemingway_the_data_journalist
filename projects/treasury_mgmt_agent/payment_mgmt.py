import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Invoice:
    """Represents a supplier invoice."""
    invoice_id: str
    supplier_id: str
    amount: float
    terms_days: int  # e.g., 30 for Net 30
    discount_rate: float = 0.0  # e.g., 2.0 for 2%
    discount_period_days: int = 0  # e.g., 10 for "2/10"

    def __post_init__(self):
        if self.discount_rate > 0 and self.discount_period_days == 0:
            raise ValueError("discount_period_days must be specified if discount_rate is > 0")
        if self.discount_period_days > self.terms_days:
            raise ValueError("discount_period_days cannot exceed terms_days")

@dataclass
class PaymentDecision:
    """Represents the recommended payment strategy for an invoice."""
    invoice_id: str
    method: str  # "Cash on Due Date", "Dynamic Discounting", "Reverse Factoring"
    savings: float # Savings compared to paying cash on due date
    effective_dpo: int
    details: Dict[str, Any] = field(default_factory=dict)


class PaymentOptimizer:
    """
    Optimizes payment strategies for supplier invoices using various payment tools
    like cash, dynamic discounting, and reverse factoring.
    """

    def __init__(self, cost_of_capital: float, reverse_factoring_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the PaymentOptimizer.

        Args:
            cost_of_capital (float): The company's annual cost of capital (e.g., 0.10 for 10%).
            reverse_factoring_config (Optional[Dict[str, Any]]): Configuration for reverse factoring.
                Example: {
                    'available': True,
                    'fee_apr': 0.05,  # 5% annual fee charged by the factor
                    'dpo_extension_days': 30 # Additional days DPO can be extended
                }
        """
        if not (0 < cost_of_capital < 1):
            raise ValueError("cost_of_capital should be a float between 0 and 1 (e.g., 0.1 for 10%)")
        self.cost_of_capital = cost_of_capital
        self.rf_config = reverse_factoring_config if reverse_factoring_config else {'available': False}
        if self.rf_config.get('available', False):
            if 'fee_apr' not in self.rf_config or 'dpo_extension_days' not in self.rf_config:
                raise ValueError("If reverse factoring is available, 'fee_apr' and 'dpo_extension_days' must be in config.")
        print(f"PaymentOptimizer initialized with Cost of Capital: {self.cost_of_capital*100:.2f}%")
        if self.rf_config.get('available'):
            print(f"Reverse Factoring enabled: Fee APR {self.rf_config['fee_apr']*100:.2f}%, DPO Extension {self.rf_config['dpo_extension_days']} days")


    def _calculate_discount_apr(self, discount_rate_percent: float, discount_period_days: int, terms_days: int) -> float:
        """
        Calculates the Annual Percentage Rate (APR) of taking a supplier discount.
        Formula: APR = (Discount % / (100% - Discount %)) * (365 / (Full Due Days - Discount Days))
        """
        if discount_rate_percent == 0 or terms_days == discount_period_days:
            return 0.0
        
        discount_decimal = discount_rate_percent / 100.0
        days_gained_by_paying_early = terms_days - discount_period_days

        if days_gained_by_paying_early <= 0: # Should not happen due to Invoice validation
            return float('inf') # Effectively, infinite return if paid on due date for discount

        apr = (discount_decimal / (1.0 - discount_decimal)) * (365.0 / days_gained_by_paying_early)
        return apr

    def evaluate_invoice_payment_options(self, invoice: Invoice) -> List[PaymentDecision]:
        """
        Evaluates different payment options for a single invoice and calculates their financial impact.
        Savings are relative to the baseline of paying cash on the original due date.
        """
        options = []

        # Option 1: Pay Cash on Due Date (Baseline)
        options.append(PaymentDecision(
            invoice_id=invoice.invoice_id,
            method="Cash on Due Date",
            savings=0.0,
            effective_dpo=invoice.terms_days,
            details={'amount_paid': invoice.amount, 'payment_day': invoice.terms_days}
        ))

        # Option 2: Dynamic Discounting
        if invoice.discount_rate > 0:
            discount_apr = self._calculate_discount_apr(invoice.discount_rate, invoice.discount_period_days, invoice.terms_days)
            
            # Net benefit of taking the discount vs. paying full amount on terms_days,
            # considering the opportunity cost of capital for paying early.
            discount_amount = invoice.amount * (invoice.discount_rate / 100.0)
            days_paid_early = invoice.terms_days - invoice.discount_period_days
            
            # Cost of using company's own funds earlier than terms_days
            opportunity_cost_of_early_payment = invoice.amount * self.cost_of_capital * (days_paid_early / 365.0)
            
            # Savings = Discount Received - Opportunity Cost of Paying Early
            # We only consider taking discount if discount_apr > self.cost_of_capital.
            # The savings calculation directly reflects this financial advantage.
            savings_dd = discount_amount - opportunity_cost_of_early_payment
            
            # If discount_apr (effective return of discount) > self.cost_of_capital (cost of funds), it's generally a good deal.
            # The savings_dd calculated above reflects this net benefit.

            options.append(PaymentDecision(
                invoice_id=invoice.invoice_id,
                method="Dynamic Discounting",
                savings=savings_dd if savings_dd > 0 else -float('inf'), # Only consider if positive net savings
                effective_dpo=invoice.discount_period_days,
                details={
                    'discount_apr': discount_apr,
                    'discount_amount': discount_amount,
                    'opportunity_cost_early_payment': opportunity_cost_of_early_payment,
                    'amount_paid': invoice.amount - discount_amount,
                    'payment_day': invoice.discount_period_days
                }
            ))
        else: # Add a placeholder if no discount is offered for consistent option list length if needed, or skip
             options.append(PaymentDecision(
                invoice_id=invoice.invoice_id,
                method="Dynamic Discounting",
                savings= -float('inf'), # Not applicable
                effective_dpo=invoice.terms_days, # No change from baseline
                details={'notes': "No discount offered"}
            ))


        # Option 3: Reverse Factoring
        if self.rf_config.get('available'):
            rf_fee_apr = self.rf_config['fee_apr']
            dpo_extension_days_rf = self.rf_config['dpo_extension_days']
            
            final_dpo_rf = invoice.terms_days + dpo_extension_days_rf
            
            # Benefit: Value of extending DPO from original terms_days using own capital
            # This is the money saved by not using own funds for 'dpo_extension_days_rf' period.
            benefit_from_dpo_extension = invoice.amount * self.cost_of_capital * (dpo_extension_days_rf / 365.0)
            
            # Cost: Fee paid to factor for providing this DPO extension.
            # Assuming the fee_apr is for the extension period.
            # If rf_fee_apr covers the entire period (original_terms + extension), this calc would change.
            # For this model, let's assume rf_fee_apr is the cost for the *additional* dpo_extension_days.
            cost_of_rf_for_extension = invoice.amount * rf_fee_apr * (dpo_extension_days_rf / 365.0)

            # A more common model is that rf_fee_apr is the rate the buyer pays the factor for the
            # entire period the factor is funding (e.g., from when supplier gets paid early until buyer pays factor).
            # Let's simplify: The buyer is essentially choosing to finance at rf_fee_apr for final_dpo_rf days
            # vs financing at self.cost_of_capital for invoice.terms_days.

            # Savings for RF = (Value of DPO with own capital for original_terms_days)
            #                  - (Cost of DPO with RF capital for final_dpo_rf days)
            # This translates to: if RF APR is lower than CoC for the extended period, it's a net gain.
            # Savings = (Benefit of DPO extension) - (Net cost of RF financing for that extension)
            # Net cost of RF for extension = Cost of RF for ext. - Benefit of using cheaper RF capital if rf_fee_apr < cost_of_capital

            # Simpler: Savings by extending DPO vs. original due date, net of RF fees for that extension.
            # The buyer effectively borrows from the factor at rf_fee_apr instead of using their own
            # cash (opportunity cost = self.cost_of_capital) for the extended period.
            savings_rf = benefit_from_dpo_extension - cost_of_rf_for_extension
            
            # Alternative: Total financing cost comparison
            # Cost of cash payment at terms_days (implicit): invoice.amount * self.cost_of_capital * (invoice.terms_days / 365.0)
            # Cost of RF payment at final_dpo_rf: invoice.amount * rf_fee_apr * (final_dpo_rf / 365.0)
            # savings_rf = (invoice.amount * self.cost_of_capital * (invoice.terms_days / 365.0)) - \
            #              (invoice.amount * rf_fee_apr * (final_dpo_rf / 365.0))
            # This would be the total difference in financing cost for the respective periods.
            # Let's stick to the clearer "benefit of extension minus cost of extension".
            
            options.append(PaymentDecision(
                invoice_id=invoice.invoice_id,
                method="Reverse Factoring",
                savings=savings_rf, # Can be negative if RF is too expensive for the extension
                effective_dpo=final_dpo_rf,
                details={
                    'rf_fee_apr': rf_fee_apr,
                    'dpo_extension_days': dpo_extension_days_rf,
                    'benefit_from_dpo_extension': benefit_from_dpo_extension,
                    'cost_of_rf_for_extension': cost_of_rf_for_extension,
                    'amount_paid_to_factor': invoice.amount, # Assuming factor pays supplier net, buyer pays gross to factor
                    'payment_day_to_factor': final_dpo_rf
                }
            ))
        else:
            options.append(PaymentDecision(
                invoice_id=invoice.invoice_id,
                method="Reverse Factoring",
                savings=-float('inf'), # Not applicable
                effective_dpo=invoice.terms_days,
                details={'notes': "Not available"}
            ))
            
        return options

    def optimize_payments(self, invoices: List[Invoice]) -> List[PaymentDecision]:
        """
        Optimizes payment for a list of invoices by selecting the best payment method for each.
        """
        optimized_decisions = []
        for invoice in invoices:
            print(f"\n--- Evaluating Invoice: {invoice.invoice_id} (Amount: {invoice.amount:.2f}) ---")
            payment_options = self.evaluate_invoice_payment_options(invoice)
            
            best_option = None
            max_savings = -float('inf')

            for option in payment_options:
                if option.method == "Dynamic Discounting" and option.details.get('discount_apr',0) == 0 and invoice.discount_rate > 0 :
                    # This means discount terms are not favorable (e.g. pay on due date for discount)
                    # or days_gained_by_paying_early is 0 making APR calc difficult to interpret simply.
                    # The savings calculation should handle this by being non-positive.
                    pass

                print(f"  Option: {option.method}, Savings: {option.savings:.2f}, Effective DPO: {option.effective_dpo} days")
                if option.savings > max_savings:
                    max_savings = option.savings
                    best_option = option
            
            if best_option:
                optimized_decisions.append(best_option)
                print(f"  BEST OPTION for {invoice.invoice_id}: {best_option.method} (Savings: {best_option.savings:.2f}, DPO: {best_option.effective_dpo})")
            else:
                # Should always have at least "Cash on Due Date"
                print(f"  ERROR: No best option found for {invoice.invoice_id}, defaulting to Cash on Due Date.")
                cash_option = next(opt for opt in payment_options if opt.method == "Cash on Due Date")
                optimized_decisions.append(cash_option)


        return optimized_decisions

if __name__ == '__main__':
    print("--- Initializing Payment Optimizer ---")
    # Scenario 1: Moderate cost of capital, RF available and reasonably priced
    optimizer1 = PaymentOptimizer(
        cost_of_capital=0.10,  # 10%
        reverse_factoring_config={
            'available': True,
            'fee_apr': 0.08,  # 8% (cheaper than CoC for extension)
            'dpo_extension_days': 30
        }
    )

    # Scenario 2: High cost of capital, RF is expensive or not as attractive
    optimizer2 = PaymentOptimizer(
        cost_of_capital=0.15,  # 15%
        reverse_factoring_config={
            'available': True,
            'fee_apr': 0.16,  # 16% (more expensive than CoC for extension)
            'dpo_extension_days': 15
        }
    )
    
    # Scenario 3: Low cost of capital, no RF
    optimizer3 = PaymentOptimizer(cost_of_capital=0.05)


    invoices_set1 = [
        Invoice(invoice_id="INV001", supplier_id="SUPA", amount=10000, terms_days=30, discount_rate=2.0, discount_period_days=10), # 2/10 Net 30
        Invoice(invoice_id="INV002", supplier_id="SUPB", amount=50000, terms_days=45), # Net 45, no discount
        Invoice(invoice_id="INV003", supplier_id="SUPC", amount=20000, terms_days=60, discount_rate=1.0, discount_period_days=15), # 1/15 Net 60
        Invoice(invoice_id="INV004", supplier_id="SUPD", amount=5000, terms_days=30, discount_rate=0.5, discount_period_days=29), # Bad discount: 0.5/29 Net 30
    ]

    print("\n--- Optimizing Payments (Scenario 1: CoC 10%, RF 8% for +30 days) ---")
    decisions1 = optimizer1.optimize_payments(invoices_set1)
    # for decision in decisions1:
    #     print(f"Invoice {decision.invoice_id}: Choose {decision.method}, Savings: {decision.savings:.2f}, DPO: {decision.effective_dpo} days. Details: {decision.details}")

    print("\n\n--- Optimizing Payments (Scenario 2: CoC 15%, RF 16% for +15 days) ---")
    decisions2 = optimizer2.optimize_payments(invoices_set1)
    # for decision in decisions2:
    #     print(f"Invoice {decision.invoice_id}: Choose {decision.method}, Savings: {decision.savings:.2f}, DPO: {decision.effective_dpo} days. Details: {decision.details}")

    print("\n\n--- Optimizing Payments (Scenario 3: CoC 5%, No RF) ---")
    invoices_set2 = [ # Slightly different set for variety
        Invoice(invoice_id="INV005", supplier_id="SUPE", amount=100000, terms_days=90, discount_rate=3.0, discount_period_days=30), # 3/30 Net 90
        Invoice(invoice_id="INV006", supplier_id="SUPF", amount=25000, terms_days=30),
    ]
    decisions3 = optimizer3.optimize_payments(invoices_set2)
    # for decision in decisions3:
    #     print(f"Invoice {decision.invoice_id}: Choose {decision.method}, Savings: {decision.savings:.2f}, DPO: {decision.effective_dpo} days. Details: {decision.details}")

    print("\n--- Payment Optimization Examples Complete ---")
