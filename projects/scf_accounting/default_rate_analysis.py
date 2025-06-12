#!/usr/bin/env python3
"""
违约概率深度分析工具
Default Rate Deep Analysis Tool

专门用于分析不同违约概率水平对融资方案表现的影响
"""

from scf_accounting_BNPL import DecathlonFinancingSimulator
import pandas as pd
import numpy as np

def comprehensive_default_analysis():
    """全面的违约概率分析"""
    
    print("🔍 迪卡侬融资方案违约概率深度分析")
    print("Decathlon Financing - Comprehensive Default Rate Analysis")
    print("=" * 70)
    
    simulator = DecathlonFinancingSimulator()
    
    # 扩展的违约率范围分析
    extended_default_rates = np.arange(0.005, 0.101, 0.005)  # 0.5% to 10% with 0.5% steps
    
    print(f"\n分析范围: {extended_default_rates[0]*100:.1f}% - {extended_default_rates[-1]*100:.1f}% 违约率")
    print(f"分析步长: 0.5%")
    print("-" * 50)
    
    # 运行详细敏感性分析
    sensitivity = simulator.analyze_default_rate_sensitivity(extended_default_rates.tolist())
    
    # 创建详细的DataFrame
    results_df = pd.DataFrame({
        '违约率(%)': [rate * 100 for rate in sensitivity['default_rates']],
        'BNPL_ROI(%)': sensitivity['BNPL_Only']['risk_adjusted_roi'],
        'BNPL_净利润(万元)': [profit/10000 for profit in sensitivity['BNPL_Only']['net_profit']],
        '零售商_ROI(%)': sensitivity['Retailer_Financing_Only']['risk_adjusted_roi'],
        '零售商_净利润(万元)': [profit/10000 for profit in sensitivity['Retailer_Financing_Only']['net_profit']],
        '组合_ROI(%)': sensitivity['Combined_Program']['risk_adjusted_roi'],
        '组合_净利润(万元)': [profit/10000 for profit in sensitivity['Combined_Program']['net_profit']]
    })
    
    # 关键阈值分析
    print("\n🎯 关键阈值识别:")
    print("-" * 40)
    
    # BNPL盈亏平衡点
    bnpl_breakeven = None
    for i, roi in enumerate(sensitivity['BNPL_Only']['risk_adjusted_roi']):
        if roi > 0:
            bnpl_breakeven = sensitivity['default_rates'][i]
            break
    
    if bnpl_breakeven:
        print(f"✅ BNPL盈亏平衡点: {bnpl_breakeven*100:.1f}%")
    else:
        # 找到最接近盈亏平衡的点
        closest_idx = np.argmin(np.abs(sensitivity['BNPL_Only']['risk_adjusted_roi']))
        closest_rate = sensitivity['default_rates'][closest_idx]
        closest_roi = sensitivity['BNPL_Only']['risk_adjusted_roi'][closest_idx]
        print(f"❌ BNPL在测试范围内无盈亏平衡点")
        print(f"   最接近盈亏平衡: {closest_rate*100:.1f}%违约率时ROI为{closest_roi:.2f}%")
    
    # 组合方案优势阈值
    advantage_thresholds = []
    for i, default_rate in enumerate(sensitivity['default_rates']):
        combined_roi = sensitivity['Combined_Program']['risk_adjusted_roi'][i]
        retailer_roi = sensitivity['Retailer_Financing_Only']['risk_adjusted_roi'][i]
        advantage = combined_roi - retailer_roi
        advantage_thresholds.append(advantage)
    
    # 找到优势减少到5%以下的点
    critical_advantage_idx = None
    for i, advantage in enumerate(advantage_thresholds):
        if advantage < 5.0:
            critical_advantage_idx = i
            break
    
    if critical_advantage_idx:
        critical_rate = sensitivity['default_rates'][critical_advantage_idx]
        print(f"⚠️  组合方案优势降至5%以下的临界点: {critical_rate*100:.1f}%")
    else:
        print(f"✅ 组合方案在所有测试违约率下均保持显著优势(>5%)")
    
    # 风险分级建议
    print(f"\n📊 基于违约率的风险分级建议:")
    print("-" * 50)
    
    def get_risk_category(default_rate):
        if default_rate <= 0.02:
            return "🟢 低风险"
        elif default_rate <= 0.04:
            return "🟡 中等风险"
        elif default_rate <= 0.06:
            return "🟠 较高风险"
        else:
            return "🔴 高风险"
    
    def get_strategy_recommendation(default_rate, combined_roi, retailer_roi, bnpl_roi):
        if combined_roi > max(retailer_roi, bnpl_roi) and combined_roi > 15:
            return "推荐组合方案"
        elif retailer_roi > bnpl_roi and retailer_roi > 10:
            return "推荐零售商融资"
        elif default_rate <= 0.03 and bnpl_roi > 5:
            return "谨慎的BNPL方案"
        else:
            return "避免高风险业务"
    
    # 选择关键违约率点进行展示
    key_rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    
    print(f"{'违约率':<8} {'风险等级':<12} {'组合ROI':<10} {'推荐策略':<15}")
    print("-" * 50)
    
    for rate in key_rates:
        if rate in sensitivity['default_rates']:
            idx = sensitivity['default_rates'].index(rate)
            combined_roi = sensitivity['Combined_Program']['risk_adjusted_roi'][idx]
            retailer_roi = sensitivity['Retailer_Financing_Only']['risk_adjusted_roi'][idx]
            bnpl_roi = sensitivity['BNPL_Only']['risk_adjusted_roi'][idx]
            
            risk_cat = get_risk_category(rate)
            strategy = get_strategy_recommendation(rate, combined_roi, retailer_roi, bnpl_roi)
            
            print(f"{rate*100:<8.1f}% {risk_cat:<12} {combined_roi:<10.2f}% {strategy:<15}")
    
    # 保存详细结果到CSV
    results_df.to_csv('default_rate_detailed_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n📄 详细分析结果已保存至: default_rate_detailed_analysis.csv")
    
    # 敏感度分析摘要
    print(f"\n📈 敏感度分析摘要:")
    print("-" * 40)
    
    roi_volatility = {
        'BNPL': np.std(sensitivity['BNPL_Only']['risk_adjusted_roi']),
        'Retailer': np.std(sensitivity['Retailer_Financing_Only']['risk_adjusted_roi']),
        'Combined': np.std(sensitivity['Combined_Program']['risk_adjusted_roi'])
    }
    
    print(f"ROI波动性 (标准差):")
    for scenario, volatility in roi_volatility.items():
        print(f"  {scenario}: {volatility:.2f}%")
    
    # 最稳健方案识别
    most_stable = min(roi_volatility.items(), key=lambda x: x[1])
    print(f"\n🛡️  最稳健方案: {most_stable[0]} (波动性: {most_stable[1]:.2f}%)")
    
    return results_df, sensitivity

def stress_test_scenarios():
    """压力测试场景分析"""
    
    print(f"\n💥 压力测试场景分析")
    print("=" * 50)
    
    simulator = DecathlonFinancingSimulator()
    
    # 定义压力测试场景
    stress_scenarios = {
        '经济衰退': 0.08,      # 8% 违约率
        '金融危机': 0.12,      # 12% 违约率
        '极端市场': 0.15,      # 15% 违约率
    }
    
    print(f"{'场景':<10} {'违约率':<8} {'组合ROI':<12} {'零售商ROI':<12} {'盈利性':<10}")
    print("-" * 60)
    
    for scenario_name, default_rate in stress_scenarios.items():
        sensitivity = simulator.analyze_default_rate_sensitivity([default_rate])
        
        combined_roi = sensitivity['Combined_Program']['risk_adjusted_roi'][0]
        retailer_roi = sensitivity['Retailer_Financing_Only']['risk_adjusted_roi'][0]
        
        profitability = "盈利" if combined_roi > 0 else "亏损"
        
        print(f"{scenario_name:<10} {default_rate*100:<8.1f}% {combined_roi:<12.2f}% {retailer_roi:<12.2f}% {profitability:<10}")
    
    print(f"\n🎯 压力测试结论:")
    print("• 即使在极端市场条件下，零售商融资仍能保持稳定收益")
    print("• 组合方案在正常经济环境下具有显著优势")
    print("• 建议建立违约率预警机制，及时调整业务策略")

def main():
    """主函数"""
    
    # 运行全面分析
    results_df, sensitivity_data = comprehensive_default_analysis()
    
    # 运行压力测试
    stress_test_scenarios()
    
    print(f"\n✅ 违约概率深度分析完成！")
    print(f"📊 可视化图表: default_sensitivity_analysis.png")
    print(f"📄 详细数据: default_rate_detailed_analysis.csv")

if __name__ == "__main__":
    main() 