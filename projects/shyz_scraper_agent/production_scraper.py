#!/usr/bin/env python3
"""
上海印章信息查询生产版爬虫
基于成功的DrissionPage方案的生产就绪版本
"""

import os
import sys
import time
import logging
import csv
import json
from typing import List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# 导入我们已经验证可工作的模块
from shyz_scraper_drission import SHYZScraperDrission, CompanyQuery, SealInfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionScraper:
    """生产环境印章信息爬虫"""
    
    def __init__(self, headless: bool = True, timeout: int = 60, max_retries: int = 3):
        """
        初始化生产爬虫
        
        Args:
            headless: 是否无头模式运行
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        self.scraper = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        
    def initialize(self):
        """初始化爬虫"""
        try:
            logger.info("初始化生产爬虫")
            self.scraper = SHYZScraperDrission(
                headless=self.headless,
                timeout=self.timeout
            )
            logger.info("爬虫初始化成功")
            
        except Exception as e:
            logger.error(f"爬虫初始化失败: {e}")
            raise
    
    def search_with_retry(self, query: CompanyQuery) -> List[SealInfo]:
        """带重试机制的查询"""
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"查询 {query.company_name} - 尝试 {attempt}/{self.max_retries}")
                
                results = self.scraper.search_company_seal(query)
                
                if results:
                    logger.info(f"✅ 查询成功，获得 {len(results)} 条记录")
                    return results
                else:
                    logger.warning(f"❌ 查询无结果")
                    
                    # 如果没有结果但没有异常，不需要重试
                    if attempt == 1:
                        return []
                    
            except Exception as e:
                last_error = e
                logger.error(f"查询失败 (尝试 {attempt}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
                    # 重新初始化scraper以防止状态问题
                    try:
                        self.scraper.close()
                        self.scraper = SHYZScraperDrission(
                            headless=self.headless,
                            timeout=self.timeout
                        )
                    except:
                        pass
        
        logger.error(f"查询 {query.company_name} 最终失败: {last_error}")
        return []
    
    def batch_search(self, queries: List[CompanyQuery], 
                    progress_callback=None) -> List[SealInfo]:
        """批量查询"""
        try:
            logger.info(f"开始批量查询 {len(queries)} 个公司")
            
            all_results = []
            success_count = 0
            error_count = 0
            
            for i, query in enumerate(queries, 1):
                try:
                    if progress_callback:
                        progress_callback(i, len(queries), query.company_name)
                    
                    logger.info(f"进度: {i}/{len(queries)} - {query.company_name}")
                    
                    results = self.search_with_retry(query)
                    all_results.extend(results)
                    
                    if results:
                        success_count += 1
                        logger.info(f"✅ {query.company_name}: {len(results)} 条记录")
                    else:
                        logger.warning(f"⚠️ {query.company_name}: 无记录")
                    
                    # 请求间隔，避免过于频繁
                    if i < len(queries):
                        time.sleep(1)
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ {query.company_name} 查询异常: {e}")
                    continue
            
            logger.info(f"批量查询完成:")
            logger.info(f"  总公司数: {len(queries)}")
            logger.info(f"  成功查询: {success_count}")
            logger.info(f"  查询异常: {error_count}")
            logger.info(f"  总记录数: {len(all_results)}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"批量查询失败: {e}")
            return []
    
    def close(self):
        """关闭爬虫"""
        try:
            if self.scraper:
                self.scraper.close()
                logger.info("爬虫已关闭")
        except Exception as e:
            logger.error(f"关闭爬虫失败: {e}")

def load_companies_from_csv(file_path: str) -> List[CompanyQuery]:
    """从CSV文件加载公司信息"""
    try:
        queries = []
        
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # 尝试检测分隔符
            sample = csvfile.read(1024)
            csvfile.seek(0)
            
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            for i, row in enumerate(reader, 1):
                try:
                    # 尝试不同的列名格式
                    company_name = (
                        row.get('company_name') or 
                        row.get('公司名称') or 
                        row.get('单位名称') or
                        row.get('企业名称') or
                        ''
                    ).strip()
                    
                    credit_code = (
                        row.get('credit_code') or
                        row.get('social_credit_code') or
                        row.get('统一社会信用代码') or
                        row.get('信用代码') or
                        ''
                    ).strip()
                    
                    if company_name and credit_code:
                        queries.append(CompanyQuery(
                            company_name=company_name,
                            credit_code=credit_code
                        ))
                    else:
                        logger.warning(f"第 {i} 行数据不完整: 公司名称='{company_name}', 信用代码='{credit_code}'")
                        
                except Exception as e:
                    logger.error(f"解析第 {i} 行失败: {e}")
                    continue
        
        logger.info(f"从 {file_path} 成功加载 {len(queries)} 个公司")
        
        if not queries:
            logger.error("未能加载任何有效的公司数据")
            logger.info("请确保CSV文件包含以下列名之一：")
            logger.info("  公司名称: company_name, 公司名称, 单位名称, 企业名称")
            logger.info("  信用代码: credit_code, social_credit_code, 统一社会信用代码, 信用代码")
        
        return queries
        
    except Exception as e:
        logger.error(f"加载CSV文件失败: {e}")
        return []

def save_results_to_csv(results: List[SealInfo], output_file: str):
    """保存结果到CSV文件"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['company_name', 'credit_code', 'seal_name', 'registration_date', 'seal_status', 'query_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入中文列标题
            chinese_headers = {
                'company_name': '公司名称',
                'credit_code': '统一社会信用代码', 
                'seal_name': '印章名称',
                'registration_date': '备案日期',
                'seal_status': '印章状态',
                'query_time': '查询时间'
            }
            writer.writerow(chinese_headers)
            
            # 写入数据
            for result in results:
                writer.writerow(asdict(result))
        
        logger.info(f"结果已保存到 {output_file}")
        
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")

def save_results_to_json(results: List[SealInfo], output_file: str):
    """保存结果到JSON文件"""
    try:
        data = {
            'query_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_records': len(results),
            'results': [asdict(result) for result in results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON结果已保存到 {output_file}")
        
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")

def create_summary_report(results: List[SealInfo], output_file: str):
    """创建查询汇总报告"""
    try:
        # 统计信息
        total_companies = len(set(result.company_name for result in results))
        total_seals = len(results)
        
        # 按公司分组
        company_stats = {}
        for result in results:
            company = result.company_name
            if company not in company_stats:
                company_stats[company] = []
            company_stats[company].append(result)
        
        # 印章类型统计
        seal_type_stats = {}
        for result in results:
            seal_type = result.seal_name
            seal_type_stats[seal_type] = seal_type_stats.get(seal_type, 0) + 1
        
        # 生成报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 上海印章信息查询汇总报告\n\n")
            f.write(f"查询时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 总体统计\n")
            f.write(f"- 查询公司数: {total_companies}\n")
            f.write(f"- 印章记录数: {total_seals}\n")
            f.write(f"- 平均每公司印章数: {total_seals/total_companies if total_companies > 0 else 0:.1f}\n\n")
            
            f.write("## 印章类型分布\n")
            for seal_type, count in sorted(seal_type_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {seal_type}: {count} 个\n")
            f.write("\n")
            
            f.write("## 公司详情\n")
            for company, seals in sorted(company_stats.items()):
                f.write(f"### {company}\n")
                f.write(f"信用代码: {seals[0].credit_code}\n")
                f.write(f"印章数量: {len(seals)}\n")
                for seal in seals:
                    f.write(f"- {seal.seal_name} ({seal.registration_date}, {seal.seal_status})\n")
                f.write("\n")
        
        logger.info(f"汇总报告已保存到 {output_file}")
        
    except Exception as e:
        logger.error(f"生成汇总报告失败: {e}")

def progress_callback(current: int, total: int, company_name: str):
    """进度回调函数"""
    percentage = (current / total) * 100
    print(f"\r进度: {current}/{total} ({percentage:.1f}%) - {company_name[:20]}", end='', flush=True)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='上海印章信息查询生产版爬虫')
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出文件前缀', default='results')
    parser.add_argument('--headless', action='store_true', help='无头模式运行')
    parser.add_argument('--timeout', type=int, default=60, help='超时时间（秒）')
    parser.add_argument('--retries', type=int, default=3, help='最大重试次数')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 生成输出文件名
    base_name = Path(args.input_file).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    csv_output = f"{args.output}_{base_name}_{timestamp}.csv"
    json_output = f"{args.output}_{base_name}_{timestamp}.json"
    report_output = f"{args.output}_{base_name}_{timestamp}_report.md"
    
    try:
        # 加载查询数据
        print(f"📁 加载输入文件: {args.input_file}")
        queries = load_companies_from_csv(args.input_file)
        
        if not queries:
            print("❌ 没有有效的查询数据")
            sys.exit(1)
        
        print(f"✅ 加载了 {len(queries)} 个公司")
        
        # 执行查询
        print(f"🔍 开始查询...")
        
        with ProductionScraper(
            headless=args.headless,
            timeout=args.timeout,
            max_retries=args.retries
        ) as scraper:
            
            results = scraper.batch_search(queries, progress_callback)
        
        print(f"\n\n🎉 查询完成!")
        print(f"📊 获得 {len(results)} 条印章记录")
        
        if results:
            # 保存结果
            print(f"💾 保存结果...")
            save_results_to_csv(results, csv_output)
            save_results_to_json(results, json_output)
            create_summary_report(results, report_output)
            
            print(f"✅ 输出文件:")
            print(f"   📄 CSV: {csv_output}")
            print(f"   📄 JSON: {json_output}")
            print(f"   📄 报告: {report_output}")
        else:
            print("⚠️ 没有获得任何印章记录")
    
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断查询")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 