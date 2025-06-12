#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海印章管理系统爬虫使用示例
Example usage of Shanghai Seal Management System Scraper
"""

from shyz_scrape_by_enterprise_name import SHYZScraper, CompanyQuery, load_queries_from_csv
import os


def example_single_query():
    """单个公司查询示例"""
    print("=== 单个公司查询示例 ===")
    
    # 创建查询对象（使用截图中的示例数据）
    query = CompanyQuery(
        company_name="鼎程（上海）金融信息服务有限公司",
        credit_code="91310104071162065Y"
    )
    
    # 创建爬虫实例
    scraper = SHYZScraper(headless=False)  # 设置为False以观察浏览器操作
    
    try:
        # 导航到查询页面
        print("正在导航到查询页面...")
        if scraper.navigate_to_query_page():
            print("成功进入查询页面!")
            
            # 查询单个公司
            print(f"正在查询公司: {query.company_name}")
            seal_infos = scraper.search_company_seal(query)
            
            if seal_infos:
                print(f"查询成功，找到 {len(seal_infos)} 条印章信息:")
                for i, seal_info in enumerate(seal_infos, 1):
                    print(f"\n印章信息 {i}:")
                    print(f"  印章名称: {seal_info.seal_name}")
                    print(f"  备案日期: {seal_info.registration_date}")
                    print(f"  印章状态: {seal_info.seal_status}")
            else:
                print("未找到相关印章信息")
        else:
            print("无法进入查询页面")
            
    except Exception as e:
        print(f"查询过程中发生错误: {e}")
    finally:
        scraper.close()


def example_batch_query():
    """批量查询示例"""
    print("\n=== 批量查询示例 ===")
    
    # 直接在代码中定义查询列表
    queries = [
        CompanyQuery(
            company_name="鼎程（上海）金融信息服务有限公司",
            credit_code="91310104071162065Y"
        ),
        CompanyQuery(
            company_name="上海某某科技有限公司",
            credit_code="91310000123456789X"
        ),
        CompanyQuery(
            company_name="上海测试股份有限公司", 
            credit_code="91310000789123456A"
        )
    ]
    
    # 创建爬虫实例
    scraper = SHYZScraper(headless=True)  # 无头模式，更快
    
    try:
        # 批量查询（不需要登录）
        print(f"开始批量查询 {len(queries)} 个公司...")
        results = scraper.batch_query(
            queries=queries,
            output_file="seal_results.csv",  # 结果保存为CSV文件
            delay=2.0  # 每次查询间隔2秒
        )
        
        print(f"\n批量查询完成!")
        print(f"总共查询到 {len(results)} 条印章信息")
        print("结果已保存到 'seal_results.csv' 文件")
        
        # 显示部分结果
        if results:
            print("\n前3条查询结果:")
            for i, seal_info in enumerate(results[:3], 1):
                print(f"\n{i}. {seal_info.company_name}")
                print(f"   印章名称: {seal_info.seal_name}")
                print(f"   备案日期: {seal_info.registration_date}")
                print(f"   印章状态: {seal_info.seal_status}")
                print(f"   查询时间: {seal_info.query_time}")
            
    except Exception as e:
        print(f"批量查询过程中发生错误: {e}")
    finally:
        scraper.close()


def example_csv_batch_query():
    """从CSV文件批量查询示例"""
    print("\n=== 从CSV文件批量查询示例 ===")
    
    csv_file = "companies_example.csv"
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"CSV文件 '{csv_file}' 不存在")
        print("请确保文件存在，或使用 companies_example.csv 作为模板")
        return
    
    # 从CSV文件加载查询列表
    print(f"正在从 '{csv_file}' 加载查询列表...")
    queries = load_queries_from_csv(csv_file)
    
    if not queries:
        print("未能从CSV文件加载到任何查询")
        return
    
    print(f"成功加载 {len(queries)} 个查询")
    
    # 创建爬虫实例
    scraper = SHYZScraper(headless=True)
    
    try:
        # 批量查询
        results = scraper.batch_query(
            queries=queries,
            output_file="csv_batch_results.json",  # 结果保存为JSON文件
            delay=1.5  # 每次查询间隔1.5秒
        )
        
        print(f"\nCSV批量查询完成!")
        print(f"总共查询到 {len(results)} 条印章信息")
        print("结果已保存到 'csv_batch_results.json' 文件")
        
    except Exception as e:
        print(f"CSV批量查询过程中发生错误: {e}")
    finally:
        scraper.close()


def main():
    """主函数"""
    print("上海印章管理系统爬虫使用示例")
    print("=" * 50)
    
    # 重要提示
    print("⚠️  重要提示:")
    print("1. 确保已安装所需依赖: uv pip install -r requirements.txt")
    print("2. 确保已安装Chrome浏览器")
    print("3. 不需要登录账号，直接使用信息查询功能")
    print("4. 请遵守相关法律法规，合规使用本工具")
    print("=" * 50)
    
    try:
        # 运行示例
        while True:
            print("\n请选择要运行的示例:")
            print("1. 单个公司查询示例")
            print("2. 批量查询示例")
            print("3. 从CSV文件批量查询示例")
            print("0. 退出")
            
            choice = input("\n请输入选择 (0-3): ").strip()
            
            if choice == "1":
                example_single_query()
            elif choice == "2":
                example_batch_query()
            elif choice == "3":
                example_csv_batch_query()
            elif choice == "0":
                print("退出程序")
                break
            else:
                print("无效选择，请重新输入")
                
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行过程中发生错误: {e}")


if __name__ == "__main__":
    main() 