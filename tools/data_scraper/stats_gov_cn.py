#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import csv
import argparse
from datetime import datetime
import sys

class StatsGovCnScraper:
    """
    Scraper for the National Bureau of Statistics of China (stats.gov.cn)
    Collects article listings from the statistical information release page
    """
    
    def __init__(self, output_dir="data/stats_gov_cn", verbose=True):
        self.base_url = "https://www.stats.gov.cn/sj/zxfb"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.articles = []
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_page(self, page_num=1):
        """Fetch a single page of article listings"""
        if page_num == 1:
            url = f"{self.base_url}/index.html"
        else:
            url = f"{self.base_url}/index_{page_num}.html"
            
        if self.verbose:
            print(f"Fetching page {page_num}: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'  # Ensure proper encoding
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page_num}: {e}")
            return None
    
    def parse_page(self, html_content):
        """Parse the HTML content and extract article information"""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'lxml')
        article_items = []
        
        # Find the article list container
        container = soup.select_one('.list-content')
        if not container:
            print("Could not find article list container")
            return []
        
        # Extract articles from the list
        articles = container.select('ul li')
        
        for article in articles:
            try:
                # Extract title and link
                title_elem = article.select_one('a')
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                relative_url = title_elem.get('href', '')
                
                # Construct full URL
                if relative_url.startswith('/'):
                    url = f"https://www.stats.gov.cn{relative_url}"
                elif relative_url.startswith('./'):
                    url = f"{self.base_url}/{relative_url[2:]}"
                elif not relative_url.startswith(('http://', 'https://')):
                    url = f"{self.base_url}/{relative_url}"
                else:
                    url = relative_url
                
                # Extract date from URL or from date element if available
                date = ""
                # Try to extract date from URL format like "./202503/t20250327_1959147.html"
                if relative_url and '/' in relative_url:
                    url_parts = relative_url.split('/')
                    for part in url_parts:
                        if part.startswith('t') and len(part) > 9:
                            date_part = part[1:9]  # Extract 20250327 from t20250327_1959147
                            try:
                                # Format date as YYYY-MM-DD
                                date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            except Exception:
                                pass
                
                # Look for date element as fallback
                date_elem = article.select_one('.list-date')
                if date_elem and not date:
                    date = date_elem.text.strip()
                
                # Determine category based on title or content
                category = self.determine_category(title)
                
                article_items.append({
                    'title': title,
                    'url': url,
                    'date': date,
                    'category': category,
                    'source': 'National Bureau of Statistics of China'
                })
                
            except Exception as e:
                print(f"Error parsing article: {e}")
        
        return article_items
    
    def determine_category(self, title):
        """Determine the category of an article based on its title"""
        title_lower = title.lower()
        
        # Economic indicators
        if any(kw in title_lower for kw in ['gdp', '国内生产总值', '经济增长', '经济运行', '季度经济']):
            return 'GDP'
        elif any(kw in title_lower for kw in ['cpi', '居民消费价格', '消费价格', '通胀', '物价']):
            return 'CPI'
        elif any(kw in title_lower for kw in ['ppi', '工业生产者', '出厂价格']):
            return 'PPI'
        elif any(kw in title_lower for kw in ['pmi', '采购经理指数', '制造业pmi', '非制造业商务活动指数']):
            return 'PMI'
        elif any(kw in title_lower for kw in ['工业', '工业企业', '规模以上工业', '工业增加值', '工业产能', '工业利润']):
            return 'Industrial Production'
        elif any(kw in title_lower for kw in ['固定资产投资', '投资增长', '固投', '资产投资']):
            return 'Fixed Asset Investment'
        elif any(kw in title_lower for kw in ['社会消费品零售总额', '零售', '消费', '商品零售', '消费市场']):
            return 'Retail Sales'
        elif any(kw in title_lower for kw in ['房地产', '商品住宅', '住宅销售', '房价', '70个大中城市商品住宅', '房屋新开工面积', '房地产开发投资']):
            return 'Real Estate'
        elif any(kw in title_lower for kw in ['进出口', '对外贸易', '进口', '出口', '贸易', '外贸', '海关']):
            return 'Trade'
        elif any(kw in title_lower for kw in ['失业率', '就业', '城镇调查失业率', '劳动力市场', '劳动人口', '劳动力']):
            return 'Employment'
        elif any(kw in title_lower for kw in ['能源', '发电量', '煤炭', '石油', '天然气', '电力', '可再生能源']):
            return 'Energy'
        elif any(kw in title_lower for kw in ['人口', '人口普查', '人口变化', '人口统计', '出生率', '死亡率', '老龄化']):
            return 'Population'
        elif any(kw in title_lower for kw in ['农业', '粮食', '农产品', '农村', '种植', '畜牧', '渔业']):
            return 'Agriculture'
        elif any(kw in title_lower for kw in ['服务业', '第三产业', '服务行业', '服务贸易']):
            return 'Services'
        elif any(kw in title_lower for kw in ['金融', '银行', '保险', '证券', '基金', '利率', '汇率']):
            return 'Finance'
        elif any(kw in title_lower for kw in ['财政', '税收', '预算', '政府收支', '国库']):
            return 'Fiscal'
        elif any(kw in title_lower for kw in ['高技术', '高新技术', '科技', '研发', 'r&d', '创新', '专利']):
            return 'High-Tech'
        elif any(kw in title_lower for kw in ['环境', '生态', '污染', '碳排放', '绿色发展']):
            return 'Environment'
        elif any(kw in title_lower for kw in ['运输', '物流', '交通', '公路', '铁路', '航空', '港口', '邮政']):
            return 'Transportation'
        elif any(kw in title_lower for kw in ['旅游', '酒店', '景区', '出行']):
            return 'Tourism'
        elif any(kw in title_lower for kw in ['教育', '学生', '学校', '高等教育', '义务教育']):
            return 'Education'
        elif any(kw in title_lower for kw in ['医疗', '卫生', '健康', '疾病', '医院', '医药']):
            return 'Healthcare'
        elif any(kw in title_lower for kw in ['社会保障', '养老', '保险', '社保', '福利']):
            return 'Social Security'
        elif any(kw in title_lower for kw in ['信息', '互联网', '电信', '软件', '信息技术', '数字经济', '大数据']):
            return 'Information Technology'
        elif any(kw in title_lower for kw in ['制造业', '装备制造', '智能制造']):
            return 'Manufacturing'
        elif any(kw in title_lower for kw in ['企业', '公司', '民营企业', '国有企业', '中小企业']):
            return 'Enterprise'
        elif any(kw in title_lower for kw in ['区域', '城市', '城镇化', '城乡', '省份', '地区']):
            return 'Regional Development'
        elif any(kw in title_lower for kw in ['居民收入', '居民消费', '居民生活', '家庭', '生活质量']):
            return 'Household Income'
        elif any(kw in title_lower for kw in ['统计公报', '年度统计', '月度统计', '季度统计']):
            return 'Statistical Report'
        else:
            return 'Other'
    
    def scrape_all_pages(self, max_pages=None, delay=1):
        """Scrape all pages of article listings"""
        page_num = 1
        total_articles = 0
        start_time = time.time()
        
        print(f"Starting to scrape articles from {self.base_url}")
        print(f"Max pages to scrape: {'Unlimited' if max_pages is None else max_pages}")
        
        while True:
            html_content = self.fetch_page(page_num)
            if not html_content:
                print(f"Failed to fetch page {page_num}, stopping.")
                break
            
            articles = self.parse_page(html_content)
            total_articles += len(articles)
            
            if not articles:
                print(f"No articles found on page {page_num}, stopping.")
                break
            
            self.articles.extend(articles)
            
            if self.verbose:
                print(f"Found {len(articles)} articles on page {page_num}. Total so far: {total_articles}")
                
                # Calculate and display progress
                elapsed_time = time.time() - start_time
                articles_per_second = total_articles / elapsed_time if elapsed_time > 0 else 0
                print(f"Progress: {articles_per_second:.2f} articles/second, elapsed time: {elapsed_time:.1f}s")
                
                # Print distribution of categories if we have articles
                if page_num % 10 == 0 and self.articles:
                    self.print_category_stats()
            
            # Check if we've reached the maximum number of pages
            if max_pages and page_num >= max_pages:
                print(f"Reached maximum number of pages ({max_pages}), stopping.")
                break
            
            page_num += 1
            
            # Sleep to avoid overloading the server
            if delay > 0:
                time.sleep(delay)
        
        # Final stats
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nScraping completed in {total_time:.2f} seconds")
        print(f"Total pages scraped: {page_num}")
        print(f"Total articles collected: {len(self.articles)}")
        
        if self.articles:
            self.print_category_stats()
    
    def print_category_stats(self):
        """Print statistics about article categories"""
        categories = {}
        for article in self.articles:
            category = article.get('category', 'Other')
            categories[category] = categories.get(category, 0) + 1
        
        print("\nCategory distribution:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.articles)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        print()
    
    def save_to_json(self):
        """Save the collected articles to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/stats_gov_cn_articles_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'total_articles': len(self.articles),
                'scrape_date': datetime.now().isoformat(),
                'articles': self.articles
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.articles)} articles to {filename}")
        return filename
    
    def save_to_csv(self):
        """Save the collected articles to a CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/stats_gov_cn_articles_{timestamp}.csv"
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'url', 'date', 'category', 'source'])
            writer.writeheader()
            writer.writerows(self.articles)
        
        print(f"Saved {len(self.articles)} articles to {filename}")
        return filename

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Scrape articles from the National Bureau of Statistics of China')
    parser.add_argument('--max-pages', type=int, default=None, help='Maximum number of pages to scrape (default: all available)')
    parser.add_argument('--output-dir', type=str, default='data/stats_gov_cn', help='Directory to store output files')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run scraper
    scraper = StatsGovCnScraper(output_dir=args.output_dir, verbose=not args.quiet)
    
    try:
        scraper.scrape_all_pages(max_pages=args.max_pages, delay=args.delay)
        
        # Save results
        if scraper.articles:
            scraper.save_to_json()
            scraper.save_to_csv()
        else:
            print("No articles were collected, skipping file output.")
    
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        
        # Save partial results if any
        if scraper.articles:
            print(f"Saving {len(scraper.articles)} articles collected so far...")
            scraper.save_to_json()
            scraper.save_to_csv()
        
        sys.exit(1)
