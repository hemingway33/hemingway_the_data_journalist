#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import glob
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import argparse
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class HousingPriceParser:
    """
    Parser for housing price data from National Bureau of Statistics of China
    Filters articles related to "大中城市商品住宅销售价格变动"
    Extracts and compiles six panel datasets for housing price indices
    """
    
    def __init__(self, input_dir="data/stats_gov_cn", output_dir="data/housing_prices", input_file=None, verbose=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_file = input_file
        self.verbose = verbose
        self.filtered_articles = []
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        # Panel datasets for housing price indices
        self.datasets = {
            "new_housing_price_index": {},        # 新建商品住宅销售价格指数
            "second_hand_price_index": {},        # 二手住宅销售价格指数
            "new_housing_category_1": {},         # 新建商品住宅销售价格分类指数（一）
            "new_housing_category_2": {},         # 新建商品住宅销售价格分类指数（二）
            "second_hand_category_1": {},         # 二手住宅销售价格分类指数（一）
            "second_hand_category_2": {}          # 二手住宅销售价格分类指数（二）
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_articles(self):
        """Load articles from the specified file or the most recent article JSON file"""
        if self.input_file:
            if not os.path.exists(self.input_file):
                print(f"Error: Input file not found: {self.input_file}")
                return []
            
            print(f"Loading articles from {self.input_file}")
            try:
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('articles', [])  # Get the articles list from the JSON
            except Exception as e:
                print(f"Error loading articles from {self.input_file}: {e}")
                return []
        else:
            json_files = glob.glob(f"{self.input_dir}/stats_gov_cn_articles_*.json")
            if not json_files:
                print(f"No article files found in {self.input_dir}")
                return []
            
            # Get the most recent file
            latest_file = max(json_files, key=os.path.getctime)
            print(f"Loading articles from {latest_file}")
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('articles', [])  # Get the articles list from the JSON
            except Exception as e:
                print(f"Error loading articles: {e}")
                return []
    
    def filter_housing_price_articles(self, articles):
        """Filter articles related to housing price changes in major cities"""
        filtered = []
        keywords = ["大中城市商品住宅销售价格", "70个大中城市住宅销售价格"]
        
        for article in articles:
            title = article.get('title', '')
            if any(keyword in title for keyword in keywords):
                filtered.append(article)
                if self.verbose:
                    print(f"Found relevant article: {title} ({article.get('date', '')})")
        
        self.filtered_articles = filtered
        return filtered
    
    def fetch_article_content(self, url):
        """Fetch the HTML content of an article"""
        if self.verbose:
            print(f"Fetching article content: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching article: {e}")
            return None
    
    def parse_article_tables(self, html_content, article_date):
        """Parse tables from article HTML and extract housing price data"""
        if not html_content:
            return
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Parse date from article title if possible, otherwise use the article date
        title_elem = soup.select_one('.article-title')
        article_title = title_elem.text.strip() if title_elem else ""
        
        # If no title found from element, try to get from meta or URL
        if not article_title:
            # Try to get from meta title
            meta_title = soup.select_one('meta[name="ArticleTitle"]')
            if meta_title:
                article_title = meta_title.get('content', '')
            
            # If still no title, get from page content
            if not article_title:
                # Look for the first h1/h2 element
                heading = soup.select_one('h1, h2')
                if heading:
                    article_title = heading.text.strip()
        
        # Extract date from title if possible (format: YYYY年MM月)
        # Try both patterns:
        # 1. "2023年10月份70个大中城市商品住宅销售价格变动情况"
        # 2. "2020年5月份70个大中城市商品住宅销售价格变动情况"
        date_match = re.search(r'(\d{4})年(\d{1,2})月份', article_title)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            
            # The data is for the month in the title, but the article may be published in the next month
            # So we'll use the month from the title for our data
            article_month = f"{year}-{month}"
        else:
            # Try another pattern without "份" character
            date_match = re.search(r'(\d{4})年(\d{1,2})月', article_title)
            if date_match:
                year = date_match.group(1)
                month = date_match.group(2).zfill(2)
                article_month = f"{year}-{month}"
            # Try to use article date (format: YYYY-MM-DD)
            elif article_date and len(article_date) >= 7:
                article_month = article_date[:7]  # Extract YYYY-MM
            else:
                print(f"Could not determine date for article: {article_title}")
                # As a last resort, try to extract date from the URL in the publication date format
                pub_date_elem = soup.select_one('.publish-date') or soup.select_one('.date')
                if pub_date_elem:
                    pub_date_text = pub_date_elem.text.strip()
                    pub_date_match = re.search(r'(\d{4})[-/年](\d{1,2})[-/月]', pub_date_text)
                    if pub_date_match:
                        year = pub_date_match.group(1)
                        month = pub_date_match.group(2).zfill(2)
                        article_month = f"{year}-{month}"
                    else:
                        # Try to extract from URL if present
                        url_elem = soup.select_one('meta[property="og:url"]')
                        if url_elem:
                            url = url_elem.get('content', '')
                            url_date_match = re.search(r't(\d{8})_', url)
                            if url_date_match:
                                date_str = url_date_match.group(1)
                                if len(date_str) >= 6:
                                    year = date_str[:4]
                                    month = date_str[4:6]
                                    article_month = f"{year}-{month}"
                                else:
                                    return
                            else:
                                return
                        else:
                            return
                else:
                    return
        
        if self.verbose:
            print(f"Parsing data for {article_month}")
        
        # Find the article content div that contains all tables
        content_div = soup.select_one('.trs_editor_view') or soup.select_one('.article-content')
        if not content_div:
            if self.verbose:
                print("Could not find article content div, using entire document")
            content_div = soup
        
        # Method 1: Try to identify tables by preceding headers
        table_type_map = self.identify_tables_by_headers(content_div)
        
        # If we couldn't identify any tables by headers, try by table position
        if not table_type_map:
            if self.verbose:
                print("Couldn't identify tables by headers, trying by position pattern")
            table_type_map = self.identify_tables_by_position(content_div)
        
        # Find all tables in the article
        tables = content_div.select('table')
        
        if self.verbose:
            print(f"Found {len(tables)} tables in the article")
        
        # Process each table
        for i, table in enumerate(tables):
            # Get the table type from our map
            dataset_key = table_type_map.get(table)
            
            if not dataset_key:
                if self.verbose:
                    print(f"Skipping table {i+1}, could not determine dataset type")
                continue
            
            if self.verbose:
                print(f"Processing table {i+1} for dataset: {dataset_key}")
            
            # Process rows in the table
            rows = table.select('tr')
            
            # Skip header rows - usually the first row contains column headers
            header_rows_to_skip = 1
            
            # For category tables, extract the category dimensions from header rows
            housing_size_dimensions = []
            if "category" in dataset_key and len(rows) >= 2:
                # Try to extract housing size dimensions from the header row
                header_row = rows[0]
                header_cells = header_row.select('td')
                
                # Skip first cell which is usually empty or "城市"
                for cell in header_cells[1:]:
                    cell_text = cell.text.strip()
                    if cell_text and cell_text != "城市" and "月同比" not in cell_text and "环比" not in cell_text:
                        housing_size_dimensions.append(cell_text)
                
                if self.verbose and housing_size_dimensions:
                    print(f"Found housing size dimensions: {housing_size_dimensions}")
                
                # If we have dimensions from the header, we may need to skip more header rows
                if housing_size_dimensions:
                    header_rows_to_skip = 2
            
            if len(rows) <= header_rows_to_skip:
                continue
            
            for row in rows[header_rows_to_skip:]:
                cells = row.select('td')
                if len(cells) < 2:
                    continue
                
                # Extract city name from first cell
                city = cells[0].text.strip()
                if not city or city == '城市':
                    continue
                
                # For regular indices (not categories), extract month-over-month value
                if "category" not in dataset_key:
                    try:
                        value = cells[1].text.strip()
                        value = float(value.replace('%', ''))  # Convert percentage to float
                    except (ValueError, IndexError):
                        value = np.nan
                    
                    # Initialize city in dataset if not present
                    if city not in self.datasets[dataset_key]:
                        self.datasets[dataset_key][city] = {}
                    
                    # Add the data point
                    self.datasets[dataset_key][city][article_month] = value
                
                # For category indices, extract values for each housing size dimension
                else:
                    # For category tables, process each dimension
                    for dim_idx, dimension in enumerate(housing_size_dimensions):
                        # Account for the first cell (city name)
                        cell_idx = dim_idx + 1
                        
                        if cell_idx < len(cells):
                            try:
                                value = cells[cell_idx].text.strip()
                                value = float(value.replace('%', ''))  # Convert percentage to float
                            except (ValueError, IndexError):
                                value = np.nan
                            
                            # Create a city-dimension key to store data
                            city_dim_key = f"{city}_{dimension}"
                            
                            # Initialize city-dimension in dataset if not present
                            if city_dim_key not in self.datasets[dataset_key]:
                                self.datasets[dataset_key][city_dim_key] = {}
                            
                            # Add the data point with dimension information
                            self.datasets[dataset_key][city_dim_key][article_month] = value
    
    def identify_tables_by_headers(self, content_div):
        """Identify tables based on preceding headers in the article"""
        # Find all table header paragraphs that indicate table types
        table_headers = content_div.select('p[style*="text-align:center"] span[style*="font-weight:bold"]')
        
        if self.verbose:
            print(f"Found {len(table_headers)} potential table headers")
        
        table_type_map = {}
        current_table_type = None
        
        # First, identify all table types from headers
        for header in table_headers:
            header_text = header.parent.text.strip()
            
            # Check if this is a table header
            if not ("表" in header_text and "70个大中城市" in header_text):
                continue
                
            # Determine which dataset this table corresponds to
            if "新建商品住宅销售价格指数" in header_text and "分类指数" not in header_text:
                current_table_type = "new_housing_price_index"
            elif "二手住宅销售价格指数" in header_text and "分类指数" not in header_text:
                current_table_type = "second_hand_price_index"
            elif "新建商品住宅销售价格分类指数" in header_text:
                if "（一）" in header_text or "90平方米及以下" in header_text:
                    current_table_type = "new_housing_category_1"
                else:
                    current_table_type = "new_housing_category_2"
            elif "二手住宅销售价格分类指数" in header_text:
                if "（一）" in header_text or "90平方米及以下" in header_text:
                    current_table_type = "second_hand_category_1"
                else:
                    current_table_type = "second_hand_category_2"
            
            if current_table_type:
                # Find the next table after this header
                next_table = header.find_next('table')
                if next_table:
                    table_type_map[next_table] = current_table_type
                    if self.verbose:
                        print(f"Identified table for {current_table_type} by header")
        
        return table_type_map
    
    def identify_tables_by_position(self, content_div):
        """Identify tables based on their position in the article
        Typically, the six tables follow this order:
        1. 新建商品住宅销售价格指数
        2. 二手住宅销售价格指数
        3. 新建商品住宅销售价格分类指数（一）
        4. 新建商品住宅销售价格分类指数（二）
        5. 二手住宅销售价格分类指数（一）
        6. 二手住宅销售价格分类指数（二）
        
        Some older articles may contain 12 tables (duplicates or alternative formats)
        """
        tables = content_div.select('table')
        
        # Map the expected order to table types
        expected_order = [
            "new_housing_price_index",
            "second_hand_price_index",
            "new_housing_category_1",
            "new_housing_category_2",
            "second_hand_category_1",
            "second_hand_category_2"
        ]
        
        table_type_map = {}
        
        # Case 1: Exactly 6 tables - standard layout
        if len(tables) == 6:
            for i, table in enumerate(tables):
                if i < len(expected_order):
                    table_type = expected_order[i]
                    table_type_map[table] = table_type
                    if self.verbose:
                        print(f"Identified table {i+1} as {table_type} by position")
        
        # Case 2: 12 tables - some older articles have 12 tables (possibly duplicates or table pairs)
        elif len(tables) == 12:
            # For 12-table format, the real assignment is more complex
            # Usually the first of each pair is the main table
            # Tables 1, 3, 5, 7, 9, 11 are often the usable tables (ignoring even-numbered)
            twelve_table_mapping = {
                0: "new_housing_price_index",       # Table 1
                2: "second_hand_price_index",       # Table 3
                4: "new_housing_category_1",        # Table 5
                6: "new_housing_category_2",        # Table 7
                8: "second_hand_category_1",        # Table 9
                10: "second_hand_category_2"        # Table 11
            }
            
            for idx, table_type in twelve_table_mapping.items():
                if idx < len(tables):
                    table_type_map[tables[idx]] = table_type
                    if self.verbose:
                        print(f"Identified table {idx+1} as {table_type} by position (12-table format)")
        
        # Case 3: Other number of tables - can't identify reliably
        else:
            if self.verbose:
                print(f"Expected 6 or 12 tables, found {len(tables)}, can't identify by position")
        
        return table_type_map
    
    def process_all_articles(self):
        """Process all filtered articles and extract housing price data"""
        if not self.filtered_articles:
            print("No relevant articles found")
            return
        
        print(f"Processing {len(self.filtered_articles)} housing price articles")
        
        for i, article in enumerate(self.filtered_articles):
            url = article.get('url')
            date = article.get('date', '')
            
            if self.verbose:
                print(f"Processing article {i+1}/{len(self.filtered_articles)}: {article.get('title')}")
            
            html_content = self.fetch_article_content(url)
            if html_content:
                self.parse_article_tables(html_content, date)
            
            # Sleep to avoid overwhelming the server
            time.sleep(1)
    
    def convert_to_dataframes(self):
        """Convert the collected data to pandas DataFrames"""
        dataframes = {}
        
        for dataset_name, city_data in self.datasets.items():
            if not city_data:
                if self.verbose:
                    print(f"No data for {dataset_name}, skipping")
                continue
            
            # Collect all unique months
            all_months = set()
            for city, months in city_data.items():
                all_months.update(months.keys())
            
            all_months = sorted(list(all_months))
            
            # Create dataframe with cities as index and months as columns
            df = pd.DataFrame(index=city_data.keys(), columns=all_months)
            
            # Fill the dataframe with values
            for city, months in city_data.items():
                for month, value in months.items():
                    df.loc[city, month] = value
            
            # For category datasets, add a column to specify housing size dimension
            if "category" in dataset_name:
                # Create a new column for housing size dimension
                df['housing_size'] = ""
                
                # Extract housing size dimension from the index
                for idx in df.index:
                    if "_" in idx:
                        city, dimension = idx.split("_", 1)
                        df.loc[idx, 'housing_size'] = dimension
            
            dataframes[dataset_name] = df
        
        return dataframes
    
    def save_datasets(self, dataframes):
        """Save the processed dataframes to Excel and CSV files"""
        if not dataframes:
            print("No data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to Excel (all sheets in one file)
        excel_file = f"{self.output_dir}/housing_price_indices_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for name, df in dataframes.items():
                sheet_name = self.get_friendly_sheet_name(name)
                
                # For category datasets, move housing_size column to the beginning for better readability
                if "category" in name and 'housing_size' in df.columns:
                    cols = df.columns.tolist()
                    cols.remove('housing_size')
                    df = df[['housing_size'] + cols]
                
                df.to_excel(writer, sheet_name=sheet_name)
        
        print(f"Saved all datasets to {excel_file}")
        
        # Save individual CSV files
        for name, df in dataframes.items():
            friendly_name = self.get_friendly_sheet_name(name, for_filename=True)
            csv_file = f"{self.output_dir}/{friendly_name}_{timestamp}.csv"
            
            # For category datasets, move housing_size column to the beginning
            if "category" in name and 'housing_size' in df.columns:
                cols = df.columns.tolist()
                cols.remove('housing_size')
                df = df[['housing_size'] + cols]
            
            df.to_csv(csv_file, encoding='utf-8')
            print(f"Saved {name} dataset to {csv_file}")
    
    def get_friendly_sheet_name(self, dataset_name, for_filename=False):
        """Convert internal dataset names to friendly Chinese names"""
        names = {
            "new_housing_price_index": "新建商品住宅销售价格指数",
            "second_hand_price_index": "二手住宅销售价格指数",
            "new_housing_category_1": "新建商品住宅销售价格分类指数(一)",
            "new_housing_category_2": "新建商品住宅销售价格分类指数(二)",
            "second_hand_category_1": "二手住宅销售价格分类指数(一)",
            "second_hand_category_2": "二手住宅销售价格分类指数(二)"
        }
        
        if for_filename:
            english_names = {
                "new_housing_price_index": "new_housing_price_index",
                "second_hand_price_index": "second_hand_price_index",
                "new_housing_category_1": "new_housing_category_1",
                "new_housing_category_2": "new_housing_category_2",
                "second_hand_category_1": "second_hand_category_1",
                "second_hand_category_2": "second_hand_category_2"
            }
            return english_names.get(dataset_name, dataset_name)
        
        return names.get(dataset_name, dataset_name)
    
    def run(self):
        """Run the entire housing price data extraction pipeline"""
        # Load articles
        articles = self.load_articles()
        if not articles:
            return False
        
        # Filter housing price articles
        filtered = self.filter_housing_price_articles(articles)
        if not filtered:
            print("No housing price articles found")
            return False
        
        # Process articles and extract data
        self.process_all_articles()
        
        # Convert data to dataframes
        dataframes = self.convert_to_dataframes()
        
        # Save the data
        self.save_datasets(dataframes)
        
        return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract housing price data from National Bureau of Statistics articles')
    parser.add_argument('--input-dir', default='data/stats_gov_cn', help='Directory containing article JSON files (default: data/stats_gov_cn)')
    parser.add_argument('--input-file', help='Specific JSON file to process (overrides input-dir if provided)')
    parser.add_argument('--output-dir', default='data/housing_prices', help='Directory to save output files (default: data/housing_prices)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    return parser.parse_args()

def load_articles(args):
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load articles from JSON file
    if args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            return []
        
        logger.info(f"Loading articles from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            articles = data.get('articles', [])  # Get the articles list from the JSON
    else:
        # Find the most recent JSON file in the input directory
        json_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json') and f.startswith('stats_gov_cn_articles_')]
        if not json_files:
            logger.error(f"No article JSON files found in {args.input_dir}")
            return []
        
        latest_file = sorted(json_files)[-1]
        json_path = os.path.join(args.input_dir, latest_file)
        
        logger.info(f"Loading articles from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            articles = data.get('articles', [])  # Get the articles list from the JSON
    
    # Filter articles with housing price related titles
    housing_price_articles = [
        article for article in articles 
        if '70个大中城市商品住宅销售价格' in article['title']
    ]
    
    logger.info(f"Found {len(housing_price_articles)} housing price articles out of {len(articles)} total articles")
    return housing_price_articles

def main():
    args = parse_arguments()
    
    start_time = time.time()
    
    # Load and process articles
    housing_price_articles = load_articles(args)
    if not housing_price_articles:
        logger.error("No housing price articles found. Exiting.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run parser
    parser = HousingPriceParser(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_file=args.input_file,
        verbose=not args.quiet
    )
    
    try:
        success = parser.run()
        if success:
            print("Housing price data extraction completed successfully")
        else:
            print("Housing price data extraction failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nHousing price data extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during housing price data extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 