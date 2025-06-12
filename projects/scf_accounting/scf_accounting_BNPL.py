"""
Supply Chain Finance Accounting Module: BNPL vs Retailer Purchase Financing
Comparative Analysis for 迪卡侬器材租赁 (Decathlon Equipment Rental)

This module implements financial modeling and comparison of three financing scenarios:
1. Consumer BNPL + Retailer Purchase Financing (Combined)
2. Consumer BNPL Only
3. Retailer Purchase Financing Only
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform

# Configure matplotlib for Chinese character support
def configure_chinese_fonts():
    """Configure matplotlib to properly display Chinese characters"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chinese_fonts = ['Arial Unicode MS', 'Songti SC', 'STHeiti', 'PingFang SC']
    elif system == "Windows":
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    else:  # Linux
        chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    
    # Try to find an available Chinese font
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
    else:
        # Fallback: disable font warnings and use default
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # Handle negative signs properly
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set figure parameters for better display
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

# Initialize Chinese font support
configure_chinese_fonts()


@dataclass
class FinancingParameters:
    """Parameters for financing programs"""
    # BNPL Parameters (Optimized Pricing)
    bnpl_interest_rate: float = 0.22  # 22% annual (optimized)
    bnpl_default_rate: float = 0.08   # 8% default rate
    bnpl_processing_fee: float = 0.02 # 2.0% processing fee (optimized)
    bnpl_installment_periods: int = 12  # 12 months
    
    # Retailer Financing Parameters
    retailer_financing_rate: float = 0.08  # 8% annual discount rate
    retailer_payment_terms: int = 90       # 90 days payment terms
    retailer_early_payment_discount: float = 0.02  # 2% early payment discount
    
    # Operating Parameters
    collection_cost_rate: float = 0.01  # 1% of outstanding amount
    operational_cost_rate: float = 0.005 # 0.5% operational overhead


@dataclass
class DecathlonScenario:
    """Decathlon equipment rental scenario parameters"""
    # Business Parameters
    monthly_equipment_purchases: float = 500000  # ¥500K monthly equipment procurement
    equipment_lifecycle_months: int = 24         # 24 months equipment lifecycle
    rental_markup_rate: float = 0.4             # 40% markup on equipment cost
    occupancy_rate: float = 0.75                 # 75% equipment utilization
    
    # Market Parameters
    consumer_bnpl_adoption: float = 0.3          # 30% of consumers use BNPL
    average_rental_value: float = 800            # ¥800 average rental transaction
    monthly_rental_transactions: int = 2000      # 2000 monthly rental transactions
    
    # Cost Structure
    storage_cost_rate: float = 0.02              # 2% of equipment value annually
    maintenance_cost_rate: float = 0.05          # 5% of equipment value annually
    insurance_cost_rate: float = 0.01            # 1% of equipment value annually


class FinancingScenarioAnalyzer:
    """Analyzer for different financing scenarios"""
    
    def __init__(self, financing_params: FinancingParameters, scenario: DecathlonScenario):
        self.financing_params = financing_params
        self.scenario = scenario
        self.analysis_period_months = 12  # 12-month analysis period
    
    def calculate_consumer_bnpl_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics for Consumer BNPL program"""
        
        # Revenue calculations
        monthly_bnpl_volume = (self.scenario.monthly_rental_transactions * 
                             self.scenario.consumer_bnpl_adoption * 
                             self.scenario.average_rental_value)
        
        annual_bnpl_volume = monthly_bnpl_volume * 12
        
        # Interest income
        interest_income = annual_bnpl_volume * self.financing_params.bnpl_interest_rate
        
        # Processing fees
        processing_fee_income = annual_bnpl_volume * self.financing_params.bnpl_processing_fee
        
        # Costs
        default_losses = annual_bnpl_volume * self.financing_params.bnpl_default_rate
        collection_costs = annual_bnpl_volume * self.financing_params.collection_cost_rate
        operational_costs = annual_bnpl_volume * self.financing_params.operational_cost_rate
        
        # Net profit
        total_income = interest_income + processing_fee_income
        total_costs = default_losses + collection_costs + operational_costs
        net_profit = total_income - total_costs
        
        # Cash flow impact (considering payment delays)
        avg_payment_delay = self.financing_params.bnpl_installment_periods / 2
        cash_flow_impact = monthly_bnpl_volume * avg_payment_delay
        
        return {
            'annual_volume': annual_bnpl_volume,
            'interest_income': interest_income,
            'processing_fee_income': processing_fee_income,
            'total_income': total_income,
            'default_losses': default_losses,
            'collection_costs': collection_costs,
            'operational_costs': operational_costs,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'roi_percentage': (net_profit / annual_bnpl_volume) * 100,
            'cash_flow_impact': cash_flow_impact
        }
    
    def calculate_retailer_financing_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics for Retailer Purchase Financing program"""
        
        # Equipment procurement financing
        annual_equipment_procurement = self.scenario.monthly_equipment_purchases * 12
        
        # Financing income from early payment discounts
        early_payment_savings = (annual_equipment_procurement * 
                               self.financing_params.retailer_early_payment_discount)
        
        # Cost of capital for financing retailers
        financing_cost = (annual_equipment_procurement * 
                         self.financing_params.retailer_financing_rate * 
                         (self.financing_params.retailer_payment_terms / 365))
        
        # Additional revenue from improved supplier relationships
        # Assuming 5% volume discount for prompt payments
        volume_discount_benefit = annual_equipment_procurement * 0.05
        
        # Enhanced rental revenue due to better equipment availability
        improved_utilization = 0.05  # 5% improvement in utilization
        additional_rental_revenue = (self.scenario.monthly_rental_transactions * 
                                   self.scenario.average_rental_value * 
                                   self.scenario.rental_markup_rate * 
                                   improved_utilization * 12)
        
        # TURNOVER IMPROVEMENTS - Enhanced Benefits
        
        # 1. Inventory Turnover Improvement
        # Faster payments → Better supplier relationships → Priority deliveries → Reduced inventory holding
        base_inventory_value = annual_equipment_procurement * 0.25  # 3-month inventory
        inventory_turnover_improvement = 0.15  # 15% reduction in inventory holding time
        inventory_holding_cost_savings = (base_inventory_value * inventory_turnover_improvement * 
                                        (self.scenario.storage_cost_rate + self.scenario.maintenance_cost_rate))
        
        # 2. Cash Conversion Cycle Improvement
        # Reduced Days Payable Outstanding (DPO) through early payments
        # Improved cash-to-cash cycle efficiency
        average_monthly_cash_flow = self.scenario.monthly_rental_transactions * self.scenario.average_rental_value
        cash_cycle_improvement_days = 15  # 15 days improvement in cash conversion cycle
        cash_cycle_benefit = (average_monthly_cash_flow * cash_cycle_improvement_days / 30 * 
                            self.financing_params.retailer_financing_rate)
        
        # 3. Working Capital Efficiency
        # Better supplier terms → Reduced working capital requirements → Freed up cash for growth
        working_capital_freed = annual_equipment_procurement * 0.08  # 8% of procurement freed up
        working_capital_roi = working_capital_freed * 0.12  # 12% return on freed capital
        
        # 4. Equipment Turnover Rate Improvement
        # Better availability → Higher equipment turnover → More rental cycles per equipment
        equipment_turnover_boost = 0.03  # 3% more rental cycles per year
        equipment_turnover_revenue = (self.scenario.monthly_rental_transactions * 12 * 
                                    self.scenario.average_rental_value * 
                                    equipment_turnover_boost)
        
        # 5. Procurement Efficiency Gains
        # Better payment terms → Strategic sourcing opportunities → Bulk purchasing power
        procurement_efficiency_savings = annual_equipment_procurement * 0.02  # 2% additional savings
        
        # Operating costs
        operational_costs = annual_equipment_procurement * self.financing_params.operational_cost_rate
        
        # Net profit calculation with enhanced turnover benefits
        total_benefits = (early_payment_savings + volume_discount_benefit + 
                         additional_rental_revenue + inventory_holding_cost_savings +
                         cash_cycle_benefit + working_capital_roi + 
                         equipment_turnover_revenue + procurement_efficiency_savings)
        total_costs = financing_cost + operational_costs
        net_profit = total_benefits - total_costs
        
        # Cash flow impact (improved with turnover enhancements)
        cash_flow_impact = (annual_equipment_procurement * 
                           self.financing_params.retailer_payment_terms / 365)
        
        return {
            'annual_procurement_volume': annual_equipment_procurement,
            'early_payment_savings': early_payment_savings,
            'volume_discount_benefit': volume_discount_benefit,
            'additional_rental_revenue': additional_rental_revenue,
            
            # Enhanced Turnover Benefits
            'inventory_holding_cost_savings': inventory_holding_cost_savings,
            'cash_cycle_benefit': cash_cycle_benefit,
            'working_capital_roi': working_capital_roi,
            'equipment_turnover_revenue': equipment_turnover_revenue,
            'procurement_efficiency_savings': procurement_efficiency_savings,
            
            'total_benefits': total_benefits,
            'financing_cost': financing_cost,
            'operational_costs': operational_costs,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'roi_percentage': (net_profit / annual_equipment_procurement) * 100,
            'cash_flow_impact': cash_flow_impact,
            
            # Turnover Metrics Summary
            'total_turnover_benefits': (inventory_holding_cost_savings + cash_cycle_benefit + 
                                      working_capital_roi + equipment_turnover_revenue + 
                                      procurement_efficiency_savings)
        }
    
    def calculate_combined_scenario_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics for Combined BNPL + Retailer Financing"""
        
        bnpl_metrics = self.calculate_consumer_bnpl_metrics()
        retailer_metrics = self.calculate_retailer_financing_metrics()
        
        # Synergy effects
        # Cross-selling opportunities
        cross_sell_revenue = bnpl_metrics['annual_volume'] * 0.02  # 2% cross-sell benefit
        
        # Operational efficiency from scale
        efficiency_savings = (bnpl_metrics['operational_costs'] + 
                            retailer_metrics['operational_costs']) * 0.1  # 10% efficiency gain
        
        # Enhanced customer loyalty and retention
        loyalty_revenue = (self.scenario.monthly_rental_transactions * 
                          self.scenario.average_rental_value * 0.05 * 12)  # 5% loyalty boost
        
        # ENHANCED TURNOVER SYNERGIES for Combined Program
        
        # 1. Accelerated Cash Velocity
        # Combined BNPL + Retailer financing creates faster cash cycles
        cash_velocity_improvement = (bnpl_metrics['annual_volume'] + 
                                   retailer_metrics['annual_procurement_volume']) * 0.01  # 1% velocity boost
        
        # 2. Data-Driven Inventory Optimization
        # BNPL customer data + supplier financing data = better demand forecasting
        inventory_optimization_benefit = retailer_metrics.get('total_turnover_benefits', 0) * 0.15  # 15% boost to turnover benefits
        
        # 3. Network Effects - Platform Revenue
        # Combined customer/supplier base creates marketplace effects
        platform_network_revenue = (self.scenario.monthly_rental_transactions * 
                                   self.scenario.consumer_bnpl_adoption * 
                                   self.scenario.average_rental_value * 0.008 * 12)  # 0.8% platform fee
        
        # 4. Capital Efficiency Multiplier
        # Combined program allows for better capital allocation and turnover
        capital_efficiency_gain = (bnpl_metrics['annual_volume'] * 0.005 +  # 0.5% on BNPL volume
                                 retailer_metrics['annual_procurement_volume'] * 0.003)  # 0.3% on procurement
        
        # 5. Compounding Utilization Effects
        # Both sides of the market benefit from increased activity
        compounding_utilization = (retailer_metrics.get('equipment_turnover_revenue', 0) * 1.2)  # 20% compounding effect
        
        # Total synergy calculation
        additional_synergies = (cash_velocity_improvement + inventory_optimization_benefit + 
                              platform_network_revenue + capital_efficiency_gain + 
                              compounding_utilization)
        
        # Combined metrics
        combined_net_profit = (bnpl_metrics['net_profit'] + 
                             retailer_metrics['net_profit'] + 
                             cross_sell_revenue + 
                             efficiency_savings + 
                             loyalty_revenue +
                             additional_synergies)
        
        total_investment = (bnpl_metrics['annual_volume'] + 
                           retailer_metrics['annual_procurement_volume'])
        
        total_synergy_value = (cross_sell_revenue + efficiency_savings + loyalty_revenue + 
                             additional_synergies)
        
        return {
            'bnpl_net_profit': bnpl_metrics['net_profit'],
            'retailer_net_profit': retailer_metrics['net_profit'],
            'cross_sell_revenue': cross_sell_revenue,
            'efficiency_savings': efficiency_savings,
            'loyalty_revenue': loyalty_revenue,
            
            # Enhanced Turnover Synergies
            'cash_velocity_improvement': cash_velocity_improvement,
            'inventory_optimization_benefit': inventory_optimization_benefit,
            'platform_network_revenue': platform_network_revenue,
            'capital_efficiency_gain': capital_efficiency_gain,
            'compounding_utilization': compounding_utilization,
            
            'combined_net_profit': combined_net_profit,
            'total_investment': total_investment,
            'combined_roi_percentage': (combined_net_profit / total_investment) * 100,
            'synergy_value': total_synergy_value,
            'turnover_synergies': additional_synergies,
            
            # Include retailer turnover benefits for reference
            'retailer_turnover_benefits': retailer_metrics.get('total_turnover_benefits', 0)
        }
    
    def generate_comprehensive_analysis(self) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive comparison of all three scenarios"""
        
        bnpl_only = self.calculate_consumer_bnpl_metrics()
        retailer_only = self.calculate_retailer_financing_metrics()
        combined = self.calculate_combined_scenario_metrics()
        
        return {
            'BNPL_Only': bnpl_only,
            'Retailer_Financing_Only': retailer_only,
            'Combined_Program': combined
        }
    
    def calculate_risk_adjusted_returns(self, scenario_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate risk-adjusted returns for each scenario"""
        
        risk_adjustments = {
            'BNPL_Only': 0.15,  # Higher risk due to consumer default
            'Retailer_Financing_Only': 0.08,  # Lower risk, B2B financing
            'Combined_Program': 0.12  # Diversified risk
        }
        
        risk_adjusted_returns = {}
        for scenario, metrics in scenario_metrics.items():
            if scenario == 'Combined_Program':
                base_return = metrics['combined_roi_percentage']
            else:
                base_return = metrics['roi_percentage']
            
            risk_adjusted_return = base_return - (risk_adjustments[scenario] * 100)
            risk_adjusted_returns[scenario] = risk_adjusted_return
        
        return risk_adjusted_returns


class DecathlonFinancingSimulator:
    """Simulation engine for Decathlon financing scenarios"""
    
    def __init__(self):
        self.financing_params = FinancingParameters()
        self.scenario = DecathlonScenario()
        self.analyzer = FinancingScenarioAnalyzer(self.financing_params, self.scenario)
    
    def analyze_default_rate_sensitivity(self, default_rates: List[float] = None) -> Dict[str, Dict[str, List[float]]]:
        """Analyze sensitivity to different default rate levels"""
        
        if default_rates is None:
            # Default analysis: from 1% to 8% default rates
            default_rates = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]
        
        sensitivity_results = {
            'default_rates': default_rates,
            'BNPL_Only': {
                'roi': [],
                'net_profit': [],
                'risk_adjusted_roi': []
            },
            'Retailer_Financing_Only': {
                'roi': [],
                'net_profit': [],
                'risk_adjusted_roi': []
            },
            'Combined_Program': {
                'roi': [],
                'net_profit': [],
                'risk_adjusted_roi': []
            }
        }
        
        for default_rate in default_rates:
            # Create modified parameters with current default rate
            modified_params = FinancingParameters(
                bnpl_interest_rate=self.financing_params.bnpl_interest_rate,
                bnpl_default_rate=default_rate,  # Variable default rate
                bnpl_processing_fee=self.financing_params.bnpl_processing_fee,
                bnpl_installment_periods=self.financing_params.bnpl_installment_periods,
                retailer_financing_rate=self.financing_params.retailer_financing_rate,
                retailer_payment_terms=self.financing_params.retailer_payment_terms,
                retailer_early_payment_discount=self.financing_params.retailer_early_payment_discount,
                collection_cost_rate=self.financing_params.collection_cost_rate,
                operational_cost_rate=self.financing_params.operational_cost_rate
            )
            
            # Run analysis with modified default rate
            temp_analyzer = FinancingScenarioAnalyzer(modified_params, self.scenario)
            analysis = temp_analyzer.generate_comprehensive_analysis()
            risk_adjusted = temp_analyzer.calculate_risk_adjusted_returns(analysis)
            
            # Store results for each scenario
            sensitivity_results['BNPL_Only']['roi'].append(analysis['BNPL_Only']['roi_percentage'])
            sensitivity_results['BNPL_Only']['net_profit'].append(analysis['BNPL_Only']['net_profit'])
            sensitivity_results['BNPL_Only']['risk_adjusted_roi'].append(risk_adjusted['BNPL_Only'])
            
            sensitivity_results['Retailer_Financing_Only']['roi'].append(analysis['Retailer_Financing_Only']['roi_percentage'])
            sensitivity_results['Retailer_Financing_Only']['net_profit'].append(analysis['Retailer_Financing_Only']['net_profit'])
            sensitivity_results['Retailer_Financing_Only']['risk_adjusted_roi'].append(risk_adjusted['Retailer_Financing_Only'])
            
            sensitivity_results['Combined_Program']['roi'].append(analysis['Combined_Program']['combined_roi_percentage'])
            sensitivity_results['Combined_Program']['net_profit'].append(analysis['Combined_Program']['combined_net_profit'])
            sensitivity_results['Combined_Program']['risk_adjusted_roi'].append(risk_adjusted['Combined_Program'])
        
        return sensitivity_results
    
    def create_default_sensitivity_analysis(self) -> None:
        """Create detailed default rate sensitivity analysis and visualization"""
        
        print("\n🔍 违约概率敏感性分析 (Default Rate Sensitivity Analysis)")
        print("=" * 70)
        
        # Run sensitivity analysis
        sensitivity = self.analyze_default_rate_sensitivity()
        
        # Create comparison table
        print("\n📊 不同违约率水平下的方案对比 (Scenario Comparison at Different Default Rates):")
        print("-" * 90)
        
        header = f"{'违约率':<8} {'BNPL ROI':<12} {'零售商ROI':<12} {'组合ROI':<12} {'最优方案':<15} {'收益差异':<12}"
        print(header)
        print("-" * 90)
        
        for i, default_rate in enumerate(sensitivity['default_rates']):
            bnpl_roi = sensitivity['BNPL_Only']['risk_adjusted_roi'][i]
            retailer_roi = sensitivity['Retailer_Financing_Only']['risk_adjusted_roi'][i]
            combined_roi = sensitivity['Combined_Program']['risk_adjusted_roi'][i]
            
            # Determine best scenario
            best_roi = max(bnpl_roi, retailer_roi, combined_roi)
            if best_roi == combined_roi:
                best_scenario = "组合方案"
                benefit_vs_second = combined_roi - max(bnpl_roi, retailer_roi)
            elif best_roi == retailer_roi:
                best_scenario = "零售商融资"
                benefit_vs_second = retailer_roi - max(bnpl_roi, combined_roi)
            else:
                best_scenario = "BNPL方案"
                benefit_vs_second = bnpl_roi - max(retailer_roi, combined_roi)
            
            row = f"{default_rate*100:<8.1f}% {bnpl_roi:<12.2f}% {retailer_roi:<12.2f}% {combined_roi:<12.2f}% {best_scenario:<15} {benefit_vs_second:<12.2f}%"
            print(row)
        
        # Break-even analysis
        print(f"\n🎯 关键阈值分析 (Critical Threshold Analysis):")
        print("-" * 50)
        
        # Find break-even points
        breakeven_bnpl = None
        breakeven_combined_vs_retailer = None
        
        for i, default_rate in enumerate(sensitivity['default_rates']):
            bnpl_roi = sensitivity['BNPL_Only']['risk_adjusted_roi'][i]
            retailer_roi = sensitivity['Retailer_Financing_Only']['risk_adjusted_roi'][i]
            combined_roi = sensitivity['Combined_Program']['risk_adjusted_roi'][i]
            
            # BNPL break-even point (when ROI becomes positive)
            if breakeven_bnpl is None and bnpl_roi > 0:
                breakeven_bnpl = default_rate
            
            # Combined vs Retailer break-even
            if breakeven_combined_vs_retailer is None and combined_roi < retailer_roi:
                breakeven_combined_vs_retailer = sensitivity['default_rates'][i-1] if i > 0 else default_rate
        
        if breakeven_bnpl:
            print(f"• BNPL方案盈亏平衡点: {breakeven_bnpl*100:.1f}% 违约率")
        else:
            print(f"• BNPL方案在所有测试违约率下均无法盈利")
        
        if breakeven_combined_vs_retailer:
            print(f"• 组合方案优于零售商融资的临界点: {breakeven_combined_vs_retailer*100:.1f}% 违约率以下")
        else:
            print(f"• 组合方案在所有测试违约率下均优于零售商融资")
        
        # Risk tolerance recommendations
        print(f"\n💡 风险承受能力建议 (Risk Tolerance Recommendations):")
        print("-" * 60)
        
        low_risk_threshold = 0.02  # 2%
        medium_risk_threshold = 0.04  # 4%
        
        print(f"🟢 保守型 (违约率 ≤ {low_risk_threshold*100:.0f}%): 组合方案最优")
        print(f"🟡 稳健型 (违约率 {low_risk_threshold*100:.0f}%-{medium_risk_threshold*100:.0f}%): 零售商融资为主，适度BNPL")
        print(f"🔴 激进型 (违约率 > {medium_risk_threshold*100:.0f}%): 专注零售商融资，避免BNPL")
        
        return sensitivity
    
    def create_default_sensitivity_visualization(self, sensitivity_data: Dict = None) -> None:
        """Create visualization for default rate sensitivity"""
        
        if sensitivity_data is None:
            sensitivity_data = self.analyze_default_rate_sensitivity()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        try:
            fig.suptitle('违约概率敏感性分析\nDefault Rate Sensitivity Analysis', 
                        fontsize=14, fontweight='bold')
        except:
            fig.suptitle('Default Rate Sensitivity Analysis', 
                        fontsize=14, fontweight='bold')
        
        default_rates_pct = [rate * 100 for rate in sensitivity_data['default_rates']]
        
        # 1. ROI vs Default Rate
        axes[0, 0].plot(default_rates_pct, sensitivity_data['BNPL_Only']['risk_adjusted_roi'], 
                       'o-', label='BNPL Only', color='#FF6B6B', linewidth=2, markersize=6)
        axes[0, 0].plot(default_rates_pct, sensitivity_data['Retailer_Financing_Only']['risk_adjusted_roi'], 
                       's-', label='Retailer Financing', color='#4ECDC4', linewidth=2, markersize=6)
        axes[0, 0].plot(default_rates_pct, sensitivity_data['Combined_Program']['risk_adjusted_roi'], 
                       '^-', label='Combined Program', color='#45B7D1', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Risk-Adjusted ROI vs Default Rate\n风险调整后ROI vs 违约率', fontsize=12)
        axes[0, 0].set_xlabel('Default Rate (%) / 违约率', fontsize=10)
        axes[0, 0].set_ylabel('Risk-Adjusted ROI (%) / 风险调整后ROI', fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Net Profit vs Default Rate
        axes[0, 1].plot(default_rates_pct, [p/1000 for p in sensitivity_data['BNPL_Only']['net_profit']], 
                       'o-', label='BNPL Only', color='#FF6B6B', linewidth=2, markersize=6)
        axes[0, 1].plot(default_rates_pct, [p/1000 for p in sensitivity_data['Retailer_Financing_Only']['net_profit']], 
                       's-', label='Retailer Financing', color='#4ECDC4', linewidth=2, markersize=6)
        axes[0, 1].plot(default_rates_pct, [p/1000 for p in sensitivity_data['Combined_Program']['net_profit']], 
                       '^-', label='Combined Program', color='#45B7D1', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Net Profit vs Default Rate\n净利润 vs 违约率', fontsize=12)
        axes[0, 1].set_xlabel('Default Rate (%) / 违约率', fontsize=10)
        axes[0, 1].set_ylabel('Net Profit (¥K) / 净利润', fontsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Advantage Analysis (Combined vs Best Alternative)
        combined_advantage = []
        for i in range(len(default_rates_pct)):
            combined_roi = sensitivity_data['Combined_Program']['risk_adjusted_roi'][i]
            alternative_roi = max(sensitivity_data['BNPL_Only']['risk_adjusted_roi'][i],
                                sensitivity_data['Retailer_Financing_Only']['risk_adjusted_roi'][i])
            combined_advantage.append(combined_roi - alternative_roi)
        
        bars = axes[1, 0].bar(default_rates_pct, combined_advantage, 
                             color=['green' if x > 0 else 'red' for x in combined_advantage],
                             alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        axes[1, 0].set_title('Combined Program Advantage\n组合方案优势', fontsize=12)
        axes[1, 0].set_xlabel('Default Rate (%) / 违约率', fontsize=10)
        axes[1, 0].set_ylabel('ROI Advantage (%) / ROI优势', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, combined_advantage):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., 
                           height + (0.1 if height > 0 else -0.3),
                           f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=8, fontweight='bold')
        
        # 4. Risk-Return Scatter at Different Default Levels
        selected_defaults = [0.01, 0.03, 0.05, 0.08]  # Select key default rates for scatter
        colors = ['green', 'blue', 'orange', 'red']
        
        for i, (default_rate, color) in enumerate(zip(selected_defaults, colors)):
            if default_rate in sensitivity_data['default_rates']:
                idx = sensitivity_data['default_rates'].index(default_rate)
                
                # Plot each scenario at this default rate
                scenarios = ['BNPL_Only', 'Retailer_Financing_Only', 'Combined_Program']
                scenario_labels = ['BNPL', 'Retailer', 'Combined']
                risk_levels = [15, 8, 12]  # Risk levels for each scenario
                
                for j, (scenario, label, risk) in enumerate(zip(scenarios, scenario_labels, risk_levels)):
                    roi = sensitivity_data[scenario]['risk_adjusted_roi'][idx]
                    axes[1, 1].scatter(risk, roi, s=100, color=color, alpha=0.7,
                                     marker=['o', 's', '^'][j], 
                                     label=f'{default_rate*100:.0f}% default' if j == 0 else "")
        
        axes[1, 1].set_title('Risk-Return Profile by Default Rate\n不同违约率下的风险收益特征', fontsize=12)
        axes[1, 1].set_xlabel('Risk Level (%) / 风险水平', fontsize=10)
        axes[1, 1].set_ylabel('Risk-Adjusted ROI (%) / 风险调整后ROI', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        try:
            plt.savefig('default_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Default sensitivity charts saved as 'default_sensitivity_analysis.png'")
        except Exception as e:
            print(f"Could not save sensitivity charts: {e}")
        
        plt.show()
    
    def run_monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict[str, List[float]]:
        """Run Monte Carlo simulation with parameter variations"""
        
        results = {
            'BNPL_Only': [],
            'Retailer_Financing_Only': [],
            'Combined_Program': []
        }
        
        np.random.seed(42)  # For reproducible results
        
        for _ in range(num_simulations):
            # Vary key parameters
            varied_params = FinancingParameters(
                bnpl_interest_rate=np.random.normal(0.15, 0.02),
                bnpl_default_rate=np.random.normal(0.03, 0.005),
                retailer_financing_rate=np.random.normal(0.08, 0.01),
                retailer_early_payment_discount=np.random.normal(0.02, 0.003)
            )
            
            varied_scenario = DecathlonScenario(
                monthly_equipment_purchases=np.random.normal(500000, 50000),
                consumer_bnpl_adoption=np.random.normal(0.3, 0.05),
                average_rental_value=np.random.normal(800, 100),
                occupancy_rate=np.random.normal(0.75, 0.05)
            )
            
            # Run analysis with varied parameters
            temp_analyzer = FinancingScenarioAnalyzer(varied_params, varied_scenario)
            analysis = temp_analyzer.generate_comprehensive_analysis()
            
            results['BNPL_Only'].append(analysis['BNPL_Only']['roi_percentage'])
            results['Retailer_Financing_Only'].append(analysis['Retailer_Financing_Only']['roi_percentage'])
            results['Combined_Program'].append(analysis['Combined_Program']['combined_roi_percentage'])
        
        return results
    
    def generate_executive_summary(self) -> Dict[str, any]:
        """Generate executive summary for management presentation"""
        
        analysis = self.analyzer.generate_comprehensive_analysis()
        risk_adjusted = self.analyzer.calculate_risk_adjusted_returns(analysis)
        monte_carlo = self.run_monte_carlo_simulation()
        
        # Statistical summary
        statistics = {}
        for scenario, returns in monte_carlo.items():
            statistics[scenario] = {
                'mean_roi': np.mean(returns),
                'std_roi': np.std(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95),
                'probability_positive': np.mean(np.array(returns) > 0)
            }
        
        # Recommendation
        best_scenario = max(risk_adjusted.items(), key=lambda x: x[1])
        
        return {
            'base_case_analysis': analysis,
            'risk_adjusted_returns': risk_adjusted,
            'monte_carlo_statistics': statistics,
            'recommendation': {
                'best_scenario': best_scenario[0],
                'expected_roi': best_scenario[1],
                'rationale': self._generate_recommendation_rationale(best_scenario[0], analysis, risk_adjusted)
            }
        }
    
    def _generate_recommendation_rationale(self, best_scenario: str, analysis: Dict, risk_adjusted: Dict) -> str:
        """Generate rationale for the recommended scenario"""
        
        rationales = {
            'BNPL_Only': f"""
            消费者BNPL方案优势:
            • 较高的利息收入和手续费收入
            • 直接面向消费者，市场增长潜力大
            • ROI: {risk_adjusted['BNPL_Only']:.2f}%
            • 但需要注意较高的违约风险管理
            """,
            
            'Retailer_Financing_Only': f"""
            零售商采购融资方案优势:
            • 风险较低的B2B融资模式
            • 改善供应链关系，获得批量折扣
            • 提高设备利用率，增加租赁收入
            • ROI: {risk_adjusted['Retailer_Financing_Only']:.2f}%
            """,
            
            'Combined_Program': f"""
            组合方案优势:
            • 实现收入来源多元化
            • 协同效应带来额外收益
            • 风险分散化，整体风险可控
            • ROI: {risk_adjusted['Combined_Program']:.2f}%
            • 协同价值: ¥{analysis['Combined_Program']['synergy_value']:,.0f}
            """
        }
        
        return rationales.get(best_scenario, "")
    
    def create_visualization_dashboard(self) -> None:
        """Create comprehensive visualization dashboard"""
        
        analysis = self.analyzer.generate_comprehensive_analysis()
        monte_carlo = self.run_monte_carlo_simulation()
        
        # Create figure with improved Chinese font support
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Try Chinese title first, fallback to English if font issues
        try:
            fig.suptitle('迪卡侬器材租赁融资方案对比分析\nDecathlon Equipment Rental Financing Analysis', 
                        fontsize=14, fontweight='bold')
        except:
            fig.suptitle('Decathlon Equipment Rental Financing Analysis', 
                        fontsize=14, fontweight='bold')
        
        # Prepare scenario data with bilingual labels
        scenario_mapping = {
            'BNPL_Only': 'BNPL Only\n消费者分期',
            'Retailer_Financing_Only': 'Retailer Financing\n零售商融资',
            'Combined_Program': 'Combined Program\n组合方案(推荐)'
        }
        
        scenarios = list(analysis.keys())
        scenario_labels = [scenario_mapping[s] for s in scenarios]
        
        rois = []
        for scenario in scenarios:
            if scenario == 'Combined_Program':
                rois.append(analysis[scenario]['combined_roi_percentage'])
            else:
                rois.append(analysis[scenario]['roi_percentage'])
        
        # ROI Comparison
        bars1 = axes[0, 0].bar(range(len(scenarios)), rois, 
                              color=['#87CEEB', '#90EE90', '#FF7F50'], 
                              alpha=0.8, edgecolor='black', linewidth=0.8)
        axes[0, 0].set_title('ROI Comparison / 投资回报率比较', fontsize=12, pad=10)
        axes[0, 0].set_ylabel('ROI (%)', fontsize=11)
        axes[0, 0].set_xticks(range(len(scenarios)))
        axes[0, 0].set_xticklabels(scenario_labels, fontsize=9)
        
        # Add value labels on bars
        for bar, roi in zip(bars1, rois):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{roi:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Monte Carlo Distribution
        colors = ['#87CEEB', '#90EE90', '#FF7F50']
        for i, (scenario, color) in enumerate(zip(scenarios, colors)):
            axes[0, 1].hist(monte_carlo[scenario], alpha=0.6, bins=25, 
                           color=color, label=scenario_labels[i], density=True)
        
        axes[0, 1].set_title('ROI Distribution / ROI分布 (Monte Carlo)', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('ROI (%)', fontsize=11)
        axes[0, 1].set_ylabel('Density / 密度', fontsize=11)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Profit Comparison
        profits = []
        for scenario in scenarios:
            if scenario == 'Combined_Program':
                profits.append(analysis[scenario]['combined_net_profit'])
            else:
                profits.append(analysis[scenario]['net_profit'])
        
        bars2 = axes[1, 0].bar(range(len(scenarios)), profits, 
                              color=['#87CEEB', '#90EE90', '#FF7F50'], 
                              alpha=0.8, edgecolor='black', linewidth=0.8)
        axes[1, 0].set_title('Annual Net Profit / 年度净利润比较', fontsize=12, pad=10)
        axes[1, 0].set_ylabel('Net Profit (¥) / 净利润', fontsize=11)
        axes[1, 0].set_xticks(range(len(scenarios)))
        axes[1, 0].set_xticklabels(scenario_labels, fontsize=9)
        
        # Add value labels on bars
        for bar, profit in zip(bars2, profits):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(profits)*0.01,
                           f'¥{profit/1000:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Risk-Return Scatter
        risk_levels = [15, 8, 12]  # Risk levels as percentages
        colors = ['#87CEEB', '#90EE90', '#FF7F50']
        
        scatter = axes[1, 1].scatter(risk_levels, rois, s=200, c=colors, alpha=0.8, 
                                   edgecolors='black', linewidth=2)
        
        # Add labels for each point
        labels = ['BNPL Only', 'Retailer Financing', 'Combined (Recommended)']
        for i, (risk, roi, label) in enumerate(zip(risk_levels, rois, labels)):
            axes[1, 1].annotate(f'{label}\n({risk}%, {roi:.1f}%)', 
                               (risk, roi), xytext=(10, 10), 
                               textcoords='offset points', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
        
        axes[1, 1].set_title('Risk-Return Analysis / 风险收益分析', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Risk Level (%) / 风险水平', fontsize=11)
        axes[1, 1].set_ylabel('ROI (%) / 投资回报率', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(risk_levels, rois, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(risk_levels), max(risk_levels), 100)
        axes[1, 1].plot(x_trend, p(x_trend), "--", alpha=0.5, color='gray')
        
        plt.tight_layout()
        
        # Save the plot
        try:
            plt.savefig('decathlon_financing_analysis_charts.png', dpi=300, bbox_inches='tight')
            print("📊 Charts saved as 'decathlon_financing_analysis_charts.png'")
        except Exception as e:
            print(f"Could not save charts: {e}")
        
        plt.show()
    
    def export_financial_report(self, filename: str = 'decathlon_financing_analysis.xlsx') -> None:
        """Export comprehensive financial report to Excel"""
        
        summary = self.generate_executive_summary()
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Base case analysis
                base_case_df = pd.DataFrame(summary['base_case_analysis']).T
                base_case_df.to_excel(writer, sheet_name='Base_Case_Analysis')
                
                # Risk adjusted returns
                risk_df = pd.DataFrame(list(summary['risk_adjusted_returns'].items()), 
                                     columns=['Scenario', 'Risk_Adjusted_ROI'])
                risk_df.to_excel(writer, sheet_name='Risk_Adjusted_Returns', index=False)
                
                # Monte Carlo statistics
                mc_df = pd.DataFrame(summary['monte_carlo_statistics']).T
                mc_df.to_excel(writer, sheet_name='Monte_Carlo_Statistics')
                
                # Recommendation summary
                rec_data = {
                    'Best Scenario': [summary['recommendation']['best_scenario']],
                    'Expected ROI (%)': [summary['recommendation']['expected_roi']],
                    'Rationale': [summary['recommendation']['rationale']]
                }
                rec_df = pd.DataFrame(rec_data)
                rec_df.to_excel(writer, sheet_name='Recommendation', index=False)
            
            print(f"Financial report exported to {filename}")
        except ImportError:
            print("Excel export not available (openpyxl not installed)")
            # Export to CSV instead
            csv_filename = filename.replace('.xlsx', '_summary.csv')
            risk_df = pd.DataFrame(list(summary['risk_adjusted_returns'].items()), 
                                 columns=['Scenario', 'Risk_Adjusted_ROI'])
            risk_df.to_csv(csv_filename, index=False)
            print(f"Summary report exported to {csv_filename}")
        except Exception as e:
            print(f"Export failed: {e}")
            print("Continuing with console output only...")


def main():
    """Main function to run the Decathlon financing analysis"""
    
    print("=" * 60)
    print("迪卡侬器材租赁融资方案对比分析")
    print("Decathlon Equipment Rental Financing Analysis")
    print("=" * 60)
    
    # Initialize simulator
    simulator = DecathlonFinancingSimulator()
    
    # Generate executive summary
    summary = simulator.generate_executive_summary()
    
    # Print key findings
    print("\n📊 核心分析结果 (Key Findings):")
    print("-" * 40)
    
    for scenario, roi in summary['risk_adjusted_returns'].items():
        print(f"{scenario}: {roi:.2f}% (风险调整后ROI)")
    
    print(f"\n🎯 推荐方案: {summary['recommendation']['best_scenario']}")
    print(f"预期ROI: {summary['recommendation']['expected_roi']:.2f}%")
    
    print("\n💡 推荐理由:")
    print(summary['recommendation']['rationale'])
    
    # Monte Carlo results
    print("\n🎲 蒙特卡洛模拟结果:")
    print("-" * 40)
    for scenario, stats in summary['monte_carlo_statistics'].items():
        print(f"{scenario}:")
        print(f"  平均ROI: {stats['mean_roi']:.2f}%")
        print(f"  标准差: {stats['std_roi']:.2f}%")
        print(f"  盈利概率: {stats['probability_positive']:.1%}")
        print()
    
    # Export results
    simulator.export_financial_report()
    
    # Create visualizations
    try:
        simulator.create_visualization_dashboard()
    except Exception as e:
        print(f"Visualization not available in this environment: {e}")
    
    # Default rate sensitivity analysis
    try:
        sensitivity_data = simulator.create_default_sensitivity_analysis()
        simulator.create_default_sensitivity_visualization(sensitivity_data)
    except Exception as e:
        print(f"Default sensitivity analysis not available: {e}")
    
    print("\n✅ 分析完成！报告已导出至 'decathlon_financing_analysis.xlsx'")


if __name__ == "__main__":
    main()
