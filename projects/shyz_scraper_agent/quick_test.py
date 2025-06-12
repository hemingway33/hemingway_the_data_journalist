#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证实际爬虫功能
Quick test script to verify actual scraping functionality
"""

from shyz_scrape_by_enterprise_name import SHYZScraper, CompanyQuery
import logging

logger = logging.getLogger(__name__)

def main():
    print("=== 快速功能测试 ===")
    print("测试查询: 上海遇圆实业有限公司")
    print("信用代码: 91310000MAE97Y0K5Y")
    print()
    
    scraper = SHYZScraper(
        headless=False,  # 显示浏览器窗口便于观察
        timeout=15
    )
    
    try:
        print("开始测试...")
        
        # 1. 导航到查询页面
        print("1. 导航到查询页面...")
        success = scraper.navigate_to_query_page()
        if success:
            print("✅ 成功进入查询页面")
        else:
            print("❌ 无法进入查询页面")
            return
        
        # 2. 执行查询
        print("2. 执行查询...")
        query = CompanyQuery(
            company_name="上海遇圆实业有限公司",
            credit_code="91310000MAE97Y0K5Y"
        )
        
        results = scraper.search_company_seal(query)
        
        if results:
            print(f"✅ 查询成功! 找到 {len(results)} 条印章记录:")
            for i, seal in enumerate(results, 1):
                print(f"  {i}. 印章名称: {seal.seal_name}")
                print(f"     备案日期: {seal.registration_date}")
                print(f"     状态: {seal.seal_status}")
                print()
        else:
            print("⚠️ 未找到印章记录，可能是:")
            print("   - 公司信息不正确")
            print("   - 该公司没有登记印章")
            print("   - 页面结构发生变化，需要调整选择器")
        
        print("3. 关闭浏览器...")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        logger.exception("详细错误信息")
    finally:
        scraper.close()
        print("测试完成!")

if __name__ == "__main__":
    main() 