#!/usr/bin/env python3
"""
Font Testing Script for Chinese Character Support
检查中文字体支持的测试脚本
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def list_chinese_fonts():
    """List available fonts that might support Chinese characters"""
    print("🔍 Checking available fonts for Chinese character support...")
    print("=" * 60)
    
    # Get all available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Common Chinese font names
    chinese_font_keywords = [
        'Arial Unicode', 'Songti', 'Heiti', 'PingFang', 'Hiragino',
        'Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'SimSun',
        'Noto Sans CJK', 'WenQuanYi', 'Source Han'
    ]
    
    system = platform.system()
    print(f"Operating System: {system}")
    print()
    
    found_fonts = []
    for font in set(available_fonts):  # Remove duplicates
        for keyword in chinese_font_keywords:
            if keyword.lower() in font.lower():
                found_fonts.append(font)
                break
    
    if found_fonts:
        print("✅ Found potential Chinese fonts:")
        for font in sorted(found_fonts):
            print(f"  - {font}")
    else:
        print("❌ No specific Chinese fonts found")
        print("📝 Recommended fonts to install:")
        if system == "Darwin":  # macOS
            print("  - Arial Unicode MS (usually pre-installed)")
            print("  - PingFang SC (usually pre-installed)")
        elif system == "Windows":
            print("  - Microsoft YaHei (usually pre-installed)")
            print("  - SimHei (usually pre-installed)")
        else:  # Linux
            print("  - sudo apt-get install fonts-noto-cjk")
            print("  - sudo apt-get install fonts-wqy-microhei")
    
    return found_fonts

def test_chinese_display():
    """Test Chinese character display in matplotlib"""
    print("\n🧪 Testing Chinese character display...")
    print("=" * 60)
    
    # Configure font
    system = platform.system()
    if system == "Darwin":  # macOS
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Songti SC']
    elif system == "Windows":
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi']
    else:  # Linux
        chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
    
    # Try to set a Chinese font
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = False
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ Using font: {font}")
            font_found = True
            break
    
    if not font_found:
        print("⚠️  No Chinese font found, using default (may show squares)")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    # Create a simple test plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Test data
    scenarios = ['BNPL Only\n消费者分期', 'Retailer Financing\n零售商融资', 'Combined\n组合方案']
    values = [13.0, 10.9, 21.1]
    colors = ['#87CEEB', '#90EE90', '#FF7F50']
    
    # Create bar chart
    bars = ax.bar(scenarios, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title
    ax.set_title('中文字体测试 / Chinese Font Test\nDecathlon Financing ROI Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('ROI (%)', fontsize=12)
    ax.set_xlabel('融资方案 / Financing Scenarios', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    
    # Try to save the plot
    try:
        plt.savefig('chinese_font_test.png', dpi=150, bbox_inches='tight')
        print("📊 Test chart saved as 'chinese_font_test.png'")
    except Exception as e:
        print(f"Could not save test chart: {e}")
    
    plt.show()

def main():
    """Main function to run font tests"""
    print("🔤 Chinese Font Support Testing for matplotlib")
    print("迪卡侬融资分析系统字体测试")
    print("=" * 60)
    
    # List available fonts
    found_fonts = list_chinese_fonts()
    
    # Test display
    test_chinese_display()
    
    print("\n📋 Summary:")
    print("-" * 30)
    if found_fonts:
        print("✅ Chinese font support appears to be available")
        print("✅ Charts should display Chinese characters properly")
    else:
        print("⚠️  Limited Chinese font support detected")
        print("💡 Consider installing additional Chinese fonts for better display")
    
    print("\n🚀 You can now run the main analysis:")
    print("   python scf_accounting_BNPL.py")
    print("   python demo_for_managers.py")

if __name__ == "__main__":
    main() 