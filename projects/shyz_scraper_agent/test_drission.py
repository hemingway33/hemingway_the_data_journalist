#!/usr/bin/env python3
"""
测试DrissionPage版本的爬虫
"""

import logging
from shyz_scraper_drission import SHYZScraperDrission, CompanyQuery

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_drission_scraper():
    """测试DrissionPage版本的爬虫"""
    print("=== 测试DrissionPage版本爬虫 ===")
    print("测试查询: 上海遇圆实业有限公司")
    print("信用代码: 91310000MAE97Y0K5Y")
    print("使用接口监听方式获取结果")
    print()
    
    # 创建爬虫实例
    scraper = SHYZScraperDrission(
        headless=False,  # 显示浏览器便于观察
        timeout=30
    )
    
    try:
        print("1. 导航到查询页面...")
        success = scraper.navigate_to_query_page()
        
        if success:
            print("✅ 成功进入查询页面")
            
            print("2. 执行查询并监听网络请求...")
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
                
                # 保存结果
                json_file = scraper.save_results(results, "json", "test_drission_results")
                csv_file = scraper.save_results(results, "csv", "test_drission_results")
                print(f"✅ 结果已保存到: {json_file} 和 {csv_file}")
                
            else:
                print("⚠️ 未找到印章记录")
                print("可能原因:")
                print("   - 该公司确实没有登记印章")
                print("   - 网站接口发生变化")
                print("   - 网络请求监听未捕获到相关数据")
                
                # 打印捕获到的网络请求信息
                if scraper.network_requests:
                    print(f"\n📊 捕获到 {len(scraper.network_requests)} 个网络请求:")
                    for i, req in enumerate(scraper.network_requests, 1):
                        print(f"  {i}. {req['method']} {req['url']}")
                        print(f"     时间: {req['timestamp']}")
                else:
                    print("\n⚠️ 未捕获到相关的网络请求")
        else:
            print("❌ 无法进入查询页面")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        logger.exception("详细错误信息")
        
    finally:
        print("\n3. 关闭浏览器...")
        scraper.close()
        print("测试完成!")

if __name__ == "__main__":
    test_drission_scraper() 