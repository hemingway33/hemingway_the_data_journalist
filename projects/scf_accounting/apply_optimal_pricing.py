#!/usr/bin/env python3
"""
应用优化BNPL定价
Apply Optimal BNPL Pricing

应用优化后的BNPL定价参数并展示改善效果
"""

from scf_accounting_BNPL import (
    DecathlonFinancingSimulator, 
    FinancingParameters, 
    DecathlonScenario,
    FinancingScenarioAnalyzer
)

def compare_pricing_scenarios():
    """对比不同定价方案"""
    
    print("🎯 BNPL定价方案对比分析")
    print("BNPL Pricing Scenario Comparison")
    print("=" * 60)
    
    scenario = DecathlonScenario()
    
    # 定义不同的定价方案
    pricing_scenarios = {
        '当前定价': FinancingParameters(
            bnpl_interest_rate=0.15,
            bnpl_default_rate=0.08,
            bnpl_processing_fee=0.01,
            bnpl_installment_periods=12
        ),
        '平衡方案': FinancingParameters(
            bnpl_interest_rate=0.22,
            bnpl_default_rate=0.08,
            bnpl_processing_fee=0.02,
            bnpl_installment_periods=12
        ),
        '保守方案': FinancingParameters(
            bnpl_interest_rate=0.20,
            bnpl_default_rate=0.08,
            bnpl_processing_fee=0.025,
            bnpl_installment_periods=12
        ),
        '激进方案': FinancingParameters(
            bnpl_interest_rate=0.25,
            bnpl_default_rate=0.08,
            bnpl_processing_fee=0.03,
            bnpl_installment_periods=12
        )
    }
    
    results = []
    
    for scenario_name, params in pricing_scenarios.items():
        analyzer = FinancingScenarioAnalyzer(params, scenario)
        analysis = analyzer.generate_comprehensive_analysis()
        risk_adjusted = analyzer.calculate_risk_adjusted_returns(analysis)
        
        # 计算有效利率
        effective_rate = params.bnpl_interest_rate + (params.bnpl_processing_fee / 1.0)
        
        # 市场竞争力评估
        if effective_rate < 0.20:
            competitiveness = "非常有竞争力"
        elif effective_rate < 0.25:
            competitiveness = "有竞争力"
        elif effective_rate < 0.30:
            competitiveness = "可接受"
        else:
            competitiveness = "竞争力较弱"
        
        results.append({
            'scenario': scenario_name,
            'interest_rate': f"{params.bnpl_interest_rate*100:.0f}%",
            'processing_fee': f"{params.bnpl_processing_fee*100:.1f}%",
            'effective_rate': f"{effective_rate*100:.1f}%",
            'bnpl_roi': risk_adjusted['BNPL_Only'],
            'combined_roi': risk_adjusted['Combined_Program'],
            'bnpl_profit': analysis['BNPL_Only']['net_profit'],
            'combined_profit': analysis['Combined_Program']['combined_net_profit'],
            'competitiveness': competitiveness
        })
    
    # 打印对比表
    print(f"\n📊 方案对比表:")
    print("-" * 100)
    print(f"{'方案':<8} {'利率':<6} {'手续费':<8} {'有效利率':<8} {'BNPL ROI':<10} {'组合ROI':<10} {'市场竞争力':<12}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['scenario']:<8} {result['interest_rate']:<6} {result['processing_fee']:<8} "
              f"{result['effective_rate']:<8} {result['bnpl_roi']:<10.2f}% {result['combined_roi']:<10.2f}% "
              f"{result['competitiveness']:<12}")
    
    # 详细收益分析
    print(f"\n💰 详细收益分析 (年净利润):")
    print("-" * 60)
    print(f"{'方案':<8} {'BNPL净利润':<15} {'组合净利润':<15} {'改善幅度':<15}")
    print("-" * 60)
    
    baseline_combined = results[0]['combined_profit']  # 当前定价作为基准
    
    for result in results:
        improvement = result['combined_profit'] - baseline_combined
        improvement_pct = (improvement / baseline_combined) * 100 if baseline_combined != 0 else 0
        
        print(f"{result['scenario']:<8} ¥{result['bnpl_profit']:>12,.0f} ¥{result['combined_profit']:>13,.0f} "
              f"{improvement_pct:>+11.1f}%")
    
    return results

def recommend_optimal_pricing():
    """推荐最优定价策略"""
    
    print(f"\n🎯 定价策略推荐")
    print("=" * 40)
    
    print(f"\n📋 推荐方案: 平衡方案")
    print("-" * 30)
    print(f"• 利率: 22%")
    print(f"• 手续费: 2.0%")
    print(f"• 有效利率: 24.0%")
    print(f"• 预期BNPL ROI: ~0.5%")
    print(f"• 预期组合ROI: ~25.2%")
    
    print(f"\n✅ 推荐理由:")
    print(f"1. 实现BNPL盈亏平衡")
    print(f"2. 市场竞争力较强")
    print(f"3. 显著提升组合ROI")
    print(f"4. 风险可控")
    
    print(f"\n⚠️  实施建议:")
    print(f"1. 分阶段调整: 先调整手续费，再调整利率")
    print(f"2. 密切监控客户反应和市场竞争")
    print(f"3. 根据实际违约率动态调整")
    print(f"4. 考虑差异化定价策略")

def apply_recommended_pricing():
    """应用推荐的定价并运行完整分析"""
    
    print(f"\n🚀 应用推荐定价并运行完整分析")
    print("=" * 50)
    
    # 创建推荐定价参数
    recommended_params = FinancingParameters(
        bnpl_interest_rate=0.22,    # 22%
        bnpl_default_rate=0.08,     # 8% (保持不变)
        bnpl_processing_fee=0.02,   # 2.0%
        bnpl_installment_periods=12  # 12月 (保持不变)
    )
    
    scenario = DecathlonScenario()
    
    # 运行完整分析
    analyzer = FinancingScenarioAnalyzer(recommended_params, scenario)
    analysis = analyzer.generate_comprehensive_analysis()
    risk_adjusted = analyzer.calculate_risk_adjusted_returns(analysis)
    
    print(f"\n📊 推荐定价方案分析结果:")
    print("-" * 40)
    print(f"BNPL方案:")
    print(f"  基础ROI: {analysis['BNPL_Only']['roi_percentage']:.2f}%")
    print(f"  风险调整后ROI: {risk_adjusted['BNPL_Only']:.2f}%")
    print(f"  年净利润: ¥{analysis['BNPL_Only']['net_profit']:,.0f}")
    
    print(f"\n组合方案:")
    print(f"  基础ROI: {analysis['Combined_Program']['combined_roi_percentage']:.2f}%")
    print(f"  风险调整后ROI: {risk_adjusted['Combined_Program']:.2f}%")
    print(f"  年净利润: ¥{analysis['Combined_Program']['combined_net_profit']:,.0f}")
    
    # 与当前定价对比
    current_params = FinancingParameters(
        bnpl_interest_rate=0.15,
        bnpl_default_rate=0.08,
        bnpl_processing_fee=0.01,
        bnpl_installment_periods=12
    )
    
    current_analyzer = FinancingScenarioAnalyzer(current_params, scenario)
    current_analysis = current_analyzer.generate_comprehensive_analysis()
    current_risk_adjusted = current_analyzer.calculate_risk_adjusted_returns(current_analysis)
    
    print(f"\n📈 改善效果:")
    print("-" * 20)
    bnpl_improvement = risk_adjusted['BNPL_Only'] - current_risk_adjusted['BNPL_Only']
    combined_improvement = risk_adjusted['Combined_Program'] - current_risk_adjusted['Combined_Program']
    profit_improvement = analysis['Combined_Program']['combined_net_profit'] - current_analysis['Combined_Program']['combined_net_profit']
    
    print(f"BNPL ROI提升: {bnpl_improvement:+.2f} 百分点")
    print(f"组合ROI提升: {combined_improvement:+.2f} 百分点")
    print(f"年利润增加: ¥{profit_improvement:+,.0f}")
    
    return recommended_params

def main():
    """主函数"""
    
    # 对比不同定价方案
    results = compare_pricing_scenarios()
    
    # 推荐最优定价
    recommend_optimal_pricing()
    
    # 应用推荐定价
    recommended_params = apply_recommended_pricing()
    
    print(f"\n✅ 定价优化分析完成!")
    print(f"💡 建议采用: 22%利率 + 2.0%手续费")

if __name__ == "__main__":
    main() 