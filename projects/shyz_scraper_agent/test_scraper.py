#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海印章管理系统爬虫测试脚本
Test script for Shanghai Seal Management System Scraper
"""

import sys
import time
import json
from datetime import datetime
from shyz_scrape_by_enterprise_name import SHYZScraper, CompanyQuery, load_queries_from_csv
from config import BASE_URL, SELECTORS, BROWSER_CONFIG


def test_browser_setup():
    """测试浏览器设置"""
    print("=== 测试浏览器设置 ===")
    
    try:
        scraper = SHYZScraper(headless=False, timeout=10)
        scraper._setup_driver()
        
        if scraper.driver:
            print("✅ 浏览器设置成功")
            
            # 测试访问网站
            print(f"正在访问网站: {BASE_URL}")
            scraper.driver.get(BASE_URL)
            time.sleep(3)
            
            # 检查页面标题
            title = scraper.driver.title
            print(f"页面标题: {title}")
            
            # 检查页面是否包含登录相关元素
            try:
                # 查找登录相关元素
                login_elements = []
                for selector_name, selector in SELECTORS.items():
                    if 'login' in selector_name or 'username' in selector_name:
                        try:
                            if isinstance(selector, list):
                                for sel in selector:
                                    elements = scraper.driver.find_elements("xpath", sel)
                                    if elements:
                                        login_elements.append(f"{selector_name}: {sel}")
                                        break
                            else:
                                elements = scraper.driver.find_elements("xpath", selector)
                                if elements:
                                    login_elements.append(f"{selector_name}: {selector}")
                        except Exception as e:
                            continue
                
                if login_elements:
                    print("✅ 找到登录相关元素:")
                    for elem in login_elements:
                        print(f"  - {elem}")
                else:
                    print("⚠️ 未找到预期的登录元素，可能需要调整选择器")
                    
            except Exception as e:
                print(f"❌ 检查登录元素时出错: {e}")
            
            print("✅ 网站访问测试完成")
            
        else:
            print("❌ 浏览器设置失败")
            return False
            
    except Exception as e:
        print(f"❌ 浏览器设置测试失败: {e}")
        return False
    finally:
        if 'scraper' in locals() and scraper.driver:
            scraper.close()
    
    return True


def test_csv_loading():
    """测试CSV文件加载"""
    print("\n=== 测试CSV文件加载 ===")
    
    try:
        # 测试加载示例CSV文件
        csv_file = "companies_example.csv"
        queries = load_queries_from_csv(csv_file)
        
        if queries:
            print(f"✅ 成功加载 {len(queries)} 个查询")
            for i, query in enumerate(queries[:3], 1):  # 只显示前3个
                print(f"  {i}. {query.company_name} ({query.credit_code})")
        else:
            print("❌ CSV文件加载失败或为空")
            return False
            
    except Exception as e:
        print(f"❌ CSV加载测试失败: {e}")
        return False
    
    return True


def test_data_structures():
    """测试数据结构"""
    print("\n=== 测试数据结构 ===")
    
    try:
        # 测试CompanyQuery
        query = CompanyQuery(
            company_name="测试公司",
            credit_code="91000000000000000X"
        )
        
        print("✅ CompanyQuery 创建成功:")
        print(f"  公司名称: {query.company_name}")
        print(f"  信用代码: {query.credit_code}")
        
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        return False
    
    return True


def test_configuration():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    
    try:
        from config import BASE_URL, SELECTORS, BROWSER_CONFIG, LOG_CONFIG
        
        print(f"✅ 基础URL: {BASE_URL}")
        print(f"✅ 选择器数量: {len(SELECTORS)}")
        print(f"✅ 浏览器配置项: {len(BROWSER_CONFIG)}")
        print(f"✅ 日志配置项: {len(LOG_CONFIG)}")
        
        # 检查关键配置
        required_selectors = ['username_input', 'password_input', 'login_submit_btn']
        missing_selectors = [sel for sel in required_selectors if sel not in SELECTORS]
        
        if missing_selectors:
            print(f"⚠️ 缺少关键选择器: {missing_selectors}")
        else:
            print("✅ 关键选择器配置完整")
            
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False
    
    return True


def test_network_connectivity():
    """测试网络连接"""
    print("\n=== 测试网络连接 ===")
    
    try:
        import requests
        
        # 创建会话
        session = requests.Session()
        session.headers.update({
            'User-Agent': BROWSER_CONFIG['user_agent']
        })
        
        print(f"正在测试连接到: {BASE_URL}")
        response = session.get(BASE_URL, timeout=10)
        
        print(f"✅ HTTP状态码: {response.status_code}")
        print(f"✅ 响应大小: {len(response.content)} 字节")
        
        # 检查响应内容
        if response.status_code == 200:
            content = response.text
            if '登录' in content or 'login' in content.lower():
                print("✅ 页面包含登录相关内容")
            else:
                print("⚠️ 页面可能不包含预期的登录内容")
        else:
            print(f"⚠️ HTTP状态码异常: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 网络连接测试失败: {e}")
        return False
    
    return True


def generate_test_report():
    """生成测试报告"""
    print("\n" + "="*50)
    print("上海印章管理系统爬虫 - 测试报告")
    print("="*50)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("网络连接测试", test_network_connectivity),
        ("配置文件测试", test_configuration),
        ("数据结构测试", test_data_structures), 
        ("CSV加载测试", test_csv_loading),
        ("浏览器设置测试", test_browser_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*50)
    print("测试结果汇总:")
    print("="*50)
    
    passed_count = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_count += 1
    
    print(f"\n总体结果: {passed_count}/{len(results)} 测试通过")
    
    if passed_count == len(results):
        print("🎉 所有测试通过！爬虫基础功能正常")
    elif passed_count >= len(results) * 0.8:
        print("⚠️ 大部分测试通过，存在一些问题需要注意")
    else:
        print("❌ 多项测试失败，请检查环境配置")
    
    # 使用提示
    print("\n" + "="*50)
    print("使用提示:")
    print("="*50)
    print("1. 运行前请确保:")
    print("   - 已安装 Chrome 浏览器")
    print("   - 已安装所需 Python 依赖")
    print("   - 网络连接正常")
    print()
    print("2. 实际使用时需要:")
    print("   - 有效的系统登录账号")
    print("   - 根据实际网站页面调整选择器配置")
    print("   - 遵守相关法律法规和网站使用条款")
    print()
    print("3. 如有问题请:")
    print("   - 查看 shyz_scraper.log 日志文件")
    print("   - 检查 config.py 中的选择器配置")
    print("   - 考虑更新浏览器和驱动程序")


def main():
    """主函数"""
    try:
        generate_test_report()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")


if __name__ == "__main__":
    main() 