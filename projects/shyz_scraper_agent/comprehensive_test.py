#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试脚本 - 测试多个公司的查询功能
Comprehensive test script for multiple company queries
"""

from shyz_scrape_by_enterprise_name import SHYZScraper, CompanyQuery

def comprehensive_test():
    """综合功能测试"""
    print("=== 综合功能测试 ===")
    
    # 测试多个公司
    test_companies = [
        CompanyQuery(
            company_name="鼎程（上海）金融信息服务有限公司",
            credit_code="91310104071162065Y"
        ),
        CompanyQuery(
            company_name="上海浦东发展银行股份有限公司",
            credit_code="91310000100001256Q"
        ),
        CompanyQuery(
            company_name="中国平安保险(集团)股份有限公司",
            credit_code="91440300100001772F"
        ),
        CompanyQuery(
            company_name="上海银行股份有限公司",
            credit_code="91310000100001106F"
        )
    ]
    
    print(f"准备测试 {len(test_companies)} 个公司")
    
    # 创建爬虫实例（无头模式以提高速度）
    scraper = SHYZScraper(headless=True, timeout=15)
    
    try:
        print("\n开始批量查询测试...")
        
        # 执行批量查询
        results = scraper.batch_query(
            queries=test_companies,
            output_file="comprehensive_test_results.csv",
            delay=3.0  # 较长的延迟以避免被限制
        )
        
        print(f"\n=== 测试结果总结 ===")
        print(f"总查询公司数: {len(test_companies)}")
        print(f"成功获取印章记录数: {len(results)}")
        
        if results:
            print(f"\n成功获取印章信息的公司:")
            company_results = {}
            for seal in results:
                if seal.company_name not in company_results:
                    company_results[seal.company_name] = []
                company_results[seal.company_name].append(seal)
            
            for company_name, seals in company_results.items():
                print(f"\n📋 {company_name}:")
                print(f"   信用代码: {seals[0].credit_code}")
                print(f"   印章数量: {len(seals)}")
                for i, seal in enumerate(seals, 1):
                    print(f"   印章{i}: {seal.seal_name} ({seal.seal_status})")
        else:
            print("\n⚠️ 所有测试公司都没有找到印章记录")
            print("这可能是因为:")
            print("1. 测试的公司确实没有在系统中登记印章")
            print("2. 查询条件不匹配")
            print("3. 系统访问限制")
        
        print(f"\n详细结果已保存到: comprehensive_test_results.csv")
        
    except Exception as e:
        print(f"❌ 综合测试过程中发生错误: {e}")
        
    finally:
        print("\n关闭浏览器...")
        scraper.close()
        print("综合测试完成!")

def single_company_detailed_test():
    """单个公司详细测试"""
    print("\n=== 单个公司详细测试 ===")
    
    # 使用一个常见的大公司进行测试
    query = CompanyQuery(
        company_name="上海浦东发展银行股份有限公司",
        credit_code="91310000100001256Q"
    )
    
    print(f"详细测试: {query.company_name}")
    print(f"信用代码: {query.credit_code}")
    
    # 使用非无头模式，可以观察整个过程
    scraper = SHYZScraper(headless=False, timeout=20)
    
    try:
        print("\n步骤1: 导航到查询页面...")
        if scraper.navigate_to_query_page():
            print("✅ 成功进入查询页面")
            
            print("\n步骤2: 执行查询...")
            results = scraper.search_company_seal(query)
            
            if results:
                print(f"✅ 查询成功！找到 {len(results)} 条印章记录:")
                for i, seal in enumerate(results, 1):
                    print(f"\n  📋 印章记录 {i}:")
                    print(f"     印章名称: {seal.seal_name}")
                    print(f"     备案日期: {seal.registration_date}")
                    print(f"     印章状态: {seal.seal_status}")
                    print(f"     查询时间: {seal.query_time}")
                    
                # 保存单个查询结果
                scraper._save_results(results, "single_company_test.json")
                print(f"\n结果已保存到: single_company_test.json")
                
            else:
                print("⚠️ 未找到印章记录")
                
        else:
            print("❌ 无法进入查询页面")
            
    except Exception as e:
        print(f"❌ 详细测试过程中发生错误: {e}")
        
    finally:
        print("\n关闭浏览器...")
        scraper.close()
        print("详细测试完成!")

def main():
    """主函数"""
    print("上海印章管理系统爬虫 - 综合功能测试")
    print("=" * 60)
    
    try:
        choice = input("\n请选择测试类型:\n1. 快速批量测试（无头模式）\n2. 单个公司详细测试（可观察过程）\n3. 两个都运行\n\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            comprehensive_test()
        elif choice == "2":
            single_company_detailed_test()
        elif choice == "3":
            comprehensive_test()
            single_company_detailed_test()
        else:
            print("无效选择，运行快速批量测试...")
            comprehensive_test()
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")

if __name__ == "__main__":
    main() 