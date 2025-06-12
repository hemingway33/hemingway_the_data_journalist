#!/usr/bin/env python3
"""
BNPL定价优化工具
BNPL Pricing Optimization Tool

优化BNPL利率和费用以获得最佳风险调整后ROI
"""

from scf_accounting_BNPL import (
    DecathlonFinancingSimulator, 
    FinancingParameters, 
    DecathlonScenario,
    FinancingScenarioAnalyzer
)
import pandas as pd
import numpy as np
from itertools import product

class BNPLPricingOptimizer:
    """BNPL定价优化器"""
    
    def __init__(self):
        self.scenario = DecathlonScenario()
        self.base_params = FinancingParameters()
    
    def optimize_bnpl_pricing(self, 
                             interest_rates: list = None,
                             processing_fees: list = None,
                             target_default_rate: float = 0.08) -> pd.DataFrame:
        """优化BNPL定价参数"""
        
        if interest_rates is None:
            # Test interest rates from 12% to 25%
            interest_rates = np.arange(0.12, 0.26, 0.01)
        
        if processing_fees is None:
            # Test processing fees from 1% to 4%
            processing_fees = np.arange(0.01, 0.045, 0.005)
        
        print(f"🔍 BNPL定价优化分析")
        print(f"Interest Rate Range: {interest_rates[0]*100:.0f}% - {interest_rates[-1]*100:.0f}%")
        print(f"Processing Fee Range: {processing_fees[0]*100:.1f}% - {processing_fees[-1]*100:.1f}%")
        print(f"Target Default Rate: {target_default_rate*100:.0f}%")
        print("=" * 60)
        
        results = []
        
        for interest_rate, processing_fee in product(interest_rates, processing_fees):
            # Create modified parameters
            test_params = FinancingParameters(
                bnpl_interest_rate=interest_rate,
                bnpl_default_rate=target_default_rate,
                bnpl_processing_fee=processing_fee,
                bnpl_installment_periods=self.base_params.bnpl_installment_periods,
                retailer_financing_rate=self.base_params.retailer_financing_rate,
                retailer_payment_terms=self.base_params.retailer_payment_terms,
                retailer_early_payment_discount=self.base_params.retailer_early_payment_discount,
                collection_cost_rate=self.base_params.collection_cost_rate,
                operational_cost_rate=self.base_params.operational_cost_rate
            )
            
            # Run analysis
            analyzer = FinancingScenarioAnalyzer(test_params, self.scenario)
            analysis = analyzer.generate_comprehensive_analysis()
            risk_adjusted = analyzer.calculate_risk_adjusted_returns(analysis)
            
            # Calculate effective interest rate
            effective_rate = interest_rate + (processing_fee / (test_params.bnpl_installment_periods / 12))
            
            # Store results
            results.append({
                'interest_rate_pct': interest_rate * 100,
                'processing_fee_pct': processing_fee * 100,
                'effective_rate_pct': effective_rate * 100,
                'bnpl_roi': analysis['BNPL_Only']['roi_percentage'],
                'bnpl_risk_adjusted_roi': risk_adjusted['BNPL_Only'],
                'bnpl_net_profit': analysis['BNPL_Only']['net_profit'],
                'combined_roi': analysis['Combined_Program']['combined_roi_percentage'],
                'combined_risk_adjusted_roi': risk_adjusted['Combined_Program'],
                'combined_net_profit': analysis['Combined_Program']['combined_net_profit']
            })
        
        return pd.DataFrame(results)
    
    def find_optimal_pricing(self, results_df: pd.DataFrame) -> dict:
        """找到最优定价策略"""
        
        # Filter for profitable BNPL scenarios
        profitable_bnpl = results_df[results_df['bnpl_risk_adjusted_roi'] > 0]
        
        optimization_results = {}
        
        if len(profitable_bnpl) > 0:
            # Best BNPL-only scenario
            best_bnpl = profitable_bnpl.loc[profitable_bnpl['bnpl_risk_adjusted_roi'].idxmax()]
            optimization_results['best_bnpl_only'] = best_bnpl
        else:
            optimization_results['best_bnpl_only'] = None
        
        # Best combined scenario
        best_combined = results_df.loc[results_df['combined_risk_adjusted_roi'].idxmax()]
        optimization_results['best_combined'] = best_combined
        
        # Most balanced scenario (good BNPL performance + high combined ROI)
        if len(profitable_bnpl) > 0:
            # Score based on both BNPL profitability and combined performance
            profitable_bnpl['balance_score'] = (
                0.4 * profitable_bnpl['bnpl_risk_adjusted_roi'] + 
                0.6 * profitable_bnpl['combined_risk_adjusted_roi']
            )
            best_balanced = profitable_bnpl.loc[profitable_bnpl['balance_score'].idxmax()]
            optimization_results['best_balanced'] = best_balanced
        else:
            optimization_results['best_balanced'] = None
        
        return optimization_results
    
    def create_pricing_recommendations(self, results_df: pd.DataFrame, 
                                     optimal_results: dict) -> None:
        """生成定价建议报告"""
        
        print(f"\n🎯 BNPL定价优化建议")
        print("=" * 50)
        
        # Current scenario performance
        current_analyzer = FinancingScenarioAnalyzer(self.base_params, self.scenario)
        current_analysis = current_analyzer.generate_comprehensive_analysis()
        current_risk_adjusted = current_analyzer.calculate_risk_adjusted_returns(current_analysis)
        
        print(f"\n📊 当前定价表现:")
        print(f"  利率: {self.base_params.bnpl_interest_rate*100:.1f}%")
        print(f"  手续费: {self.base_params.bnpl_processing_fee*100:.1f}%")
        print(f"  BNPL风险调整后ROI: {current_risk_adjusted['BNPL_Only']:.2f}%")
        print(f"  组合风险调整后ROI: {current_risk_adjusted['Combined_Program']:.2f}%")
        
        # Optimal scenarios
        print(f"\n🏆 优化后定价方案:")
        print("-" * 40)
        
        if optimal_results['best_bnpl_only'] is not None:
            best_bnpl = optimal_results['best_bnpl_only']
            print(f"\n1️⃣ 最佳BNPL单独方案:")
            print(f"   利率: {best_bnpl['interest_rate_pct']:.1f}%")
            print(f"   手续费: {best_bnpl['processing_fee_pct']:.1f}%")
            print(f"   有效利率: {best_bnpl['effective_rate_pct']:.1f}%")
            print(f"   BNPL风险调整后ROI: {best_bnpl['bnpl_risk_adjusted_roi']:.2f}%")
            print(f"   年净利润: ¥{best_bnpl['bnpl_net_profit']:,.0f}")
        else:
            print(f"\n1️⃣ 最佳BNPL单独方案: 无盈利方案")
        
        best_combined = optimal_results['best_combined']
        print(f"\n2️⃣ 最佳组合方案:")
        print(f"   利率: {best_combined['interest_rate_pct']:.1f}%")
        print(f"   手续费: {best_combined['processing_fee_pct']:.1f}%")
        print(f"   有效利率: {best_combined['effective_rate_pct']:.1f}%")
        print(f"   组合风险调整后ROI: {best_combined['combined_risk_adjusted_roi']:.2f}%")
        print(f"   年净利润: ¥{best_combined['combined_net_profit']:,.0f}")
        
        if optimal_results['best_balanced'] is not None:
            best_balanced = optimal_results['best_balanced']
            print(f"\n3️⃣ 最佳平衡方案 (推荐):")
            print(f"   利率: {best_balanced['interest_rate_pct']:.1f}%")
            print(f"   手续费: {best_balanced['processing_fee_pct']:.1f}%")
            print(f"   有效利率: {best_balanced['effective_rate_pct']:.1f}%")
            print(f"   BNPL风险调整后ROI: {best_balanced['bnpl_risk_adjusted_roi']:.2f}%")
            print(f"   组合风险调整后ROI: {best_balanced['combined_risk_adjusted_roi']:.2f}%")
            print(f"   年净利润: ¥{best_balanced['combined_net_profit']:,.0f}")
            
            # Calculate improvement
            bnpl_improvement = best_balanced['bnpl_risk_adjusted_roi'] - current_risk_adjusted['BNPL_Only']
            combined_improvement = best_balanced['combined_risk_adjusted_roi'] - current_risk_adjusted['Combined_Program']
            
            print(f"\n📈 改善幅度:")
            print(f"   BNPL ROI提升: {bnpl_improvement:+.2f} 百分点")
            print(f"   组合ROI提升: {combined_improvement:+.2f} 百分点")
        
        # Market competitiveness analysis
        print(f"\n🏪 市场竞争力分析:")
        print("-" * 30)
        
        if optimal_results['best_balanced'] is not None:
            effective_rate = optimal_results['best_balanced']['effective_rate_pct']
            if effective_rate < 20:
                competitiveness = "非常有竞争力"
            elif effective_rate < 25:
                competitiveness = "有竞争力"
            elif effective_rate < 30:
                competitiveness = "可接受"
            else:
                competitiveness = "竞争力较弱"
            
            print(f"   有效利率: {effective_rate:.1f}%")
            print(f"   竞争力评估: {competitiveness}")
            
            # Risk tolerance
            if optimal_results['best_balanced']['bnpl_risk_adjusted_roi'] > 5:
                risk_assessment = "低风险高收益"
            elif optimal_results['best_balanced']['bnpl_risk_adjusted_roi'] > 0:
                risk_assessment = "可接受风险"
            else:
                risk_assessment = "高风险"
            
            print(f"   风险评估: {risk_assessment}")
    
    def export_optimization_results(self, results_df: pd.DataFrame) -> None:
        """导出优化结果"""
        
        # Save detailed results
        results_df.to_csv('bnpl_pricing_optimization_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n📄 详细优化结果已保存至: bnpl_pricing_optimization_results.csv")
        
        # Create summary of top 10 scenarios
        top_combined = results_df.nlargest(10, 'combined_risk_adjusted_roi')
        top_combined.to_csv('bnpl_top_10_scenarios.csv', index=False, encoding='utf-8-sig')
        print(f"📄 前10优化方案已保存至: bnpl_top_10_scenarios.csv")

def quick_pricing_test():
    """快速定价测试"""
    
    print("🚀 BNPL快速定价测试")
    print("=" * 40)
    
    optimizer = BNPLPricingOptimizer()
    
    # Test a few key scenarios
    test_scenarios = [
        (0.18, 0.025),  # 18% + 2.5%
        (0.20, 0.02),   # 20% + 2.0%
        (0.22, 0.015),  # 22% + 1.5%
        (0.24, 0.01),   # 24% + 1.0%
    ]
    
    results = []
    for interest_rate, processing_fee in test_scenarios:
        test_params = FinancingParameters(
            bnpl_interest_rate=interest_rate,
            bnpl_default_rate=0.08,  # Current default rate
            bnpl_processing_fee=processing_fee,
            bnpl_installment_periods=12
        )
        
        analyzer = FinancingScenarioAnalyzer(test_params, DecathlonScenario())
        analysis = analyzer.generate_comprehensive_analysis()
        risk_adjusted = analyzer.calculate_risk_adjusted_returns(analysis)
        
        effective_rate = interest_rate + (processing_fee / 1.0)  # Annual equivalent
        
        results.append({
            'scenario': f"{interest_rate*100:.0f}% + {processing_fee*100:.1f}%",
            'effective_rate': f"{effective_rate*100:.1f}%",
            'bnpl_roi': f"{risk_adjusted['BNPL_Only']:.2f}%",
            'combined_roi': f"{risk_adjusted['Combined_Program']:.2f}%"
        })
    
    print(f"\n{'方案':<15} {'有效利率':<10} {'BNPL ROI':<12} {'组合ROI':<12}")
    print("-" * 55)
    for result in results:
        print(f"{result['scenario']:<15} {result['effective_rate']:<10} {result['bnpl_roi']:<12} {result['combined_roi']:<12}")

def main():
    """主函数"""
    
    # Quick test first
    quick_pricing_test()
    
    print(f"\n\n" + "="*60)
    
    # Full optimization
    optimizer = BNPLPricingOptimizer()
    
    # Run optimization with reasonable ranges
    results_df = optimizer.optimize_bnpl_pricing(
        interest_rates=np.arange(0.16, 0.26, 0.01),  # 16% to 25%
        processing_fees=np.arange(0.01, 0.035, 0.005),  # 1% to 3%
        target_default_rate=0.08
    )
    
    # Find optimal pricing
    optimal_results = optimizer.find_optimal_pricing(results_df)
    
    # Create recommendations
    optimizer.create_pricing_recommendations(results_df, optimal_results)
    
    # Export results
    optimizer.export_optimization_results(results_df)
    
    print(f"\n✅ BNPL定价优化完成！")

if __name__ == "__main__":
    main() 