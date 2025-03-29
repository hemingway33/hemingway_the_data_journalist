# Housing Price Data Extractor

This tool extracts and compiles housing price data from the National Bureau of Statistics of China's website, specifically from articles about "70个大中城市商品住宅销售价格变动情况" (Housing Price Changes in 70 Major Cities).

## Features

- Filters relevant housing price articles from the scraped data
- Extracts data from six different types of housing price indices
- Captures housing size dimensions for category indices (e.g., "90平方米及以下", "90-144平方米", etc.)
- Compiles the data into panel datasets with cities as rows and months as columns
- Saves the data in both Excel (all indices in separate sheets) and CSV format

## Datasets

The tool extracts the following six datasets:

1. **新建商品住宅销售价格指数** (New Housing Price Index)
   - Monthly changes in prices of newly constructed residential properties

2. **二手住宅销售价格指数** (Second-hand Housing Price Index)
   - Monthly changes in prices of second-hand residential properties

3. **新建商品住宅销售价格分类指数（一）** (New Housing Category Index 1)
   - Price index for newly constructed residential properties by housing size
   - Includes dimensions like "90平方米及以下" (≤90m²), "90-144平方米" (90-144m²), and "144平方米以上" (>144m²)

4. **新建商品住宅销售价格分类指数（二）** (New Housing Category Index 2)
   - Price index for newly constructed residential properties by another categorization
   - May include price tiers or other categorizations

5. **二手住宅销售价格分类指数（一）** (Second-hand Housing Category Index 1)
   - Price index for second-hand residential properties by housing size
   - Similar dimensions as new housing category index 1

6. **二手住宅销售价格分类指数（二）** (Second-hand Housing Category Index 2)
   - Price index for second-hand residential properties by another categorization
   - Similar dimensions as new housing category index 2

## Usage

```bash
python housing_price_parser.py [options]
```

### Options:

- `--input-dir DIR`: Directory containing article JSON files (default: data/stats_gov_cn)
- `--output-dir DIR`: Directory to save output files (default: data/housing_prices)
- `--quiet`: Suppress detailed output

### Prerequisites:

This tool depends on the output from the stats_gov_cn.py scraper. You should run the stats_gov_cn.py scraper first to collect article listings.

### Example:

```bash
# First, scrape articles from stats.gov.cn
python stats_gov_cn.py

# Then extract housing price data
python housing_price_parser.py
```

## Output Files

The tool generates the following output files:

1. `housing_price_indices_YYYYMMDD_HHMMSS.xlsx`: Excel file containing all six datasets in separate sheets
2. `new_housing_price_index_YYYYMMDD_HHMMSS.csv`: CSV file for new housing price index
3. `second_hand_price_index_YYYYMMDD_HHMMSS.csv`: CSV file for second-hand housing price index
4. `new_housing_category_1_YYYYMMDD_HHMMSS.csv`: CSV file for new housing category index 1 (with housing size dimensions)
5. `new_housing_category_2_YYYYMMDD_HHMMSS.csv`: CSV file for new housing category index 2 (with housing size dimensions)
6. `second_hand_category_1_YYYYMMDD_HHMMSS.csv`: CSV file for second-hand category index 1 (with housing size dimensions)
7. `second_hand_category_2_YYYYMMDD_HHMMSS.csv`: CSV file for second-hand category index 2 (with housing size dimensions)

## Data Format

The data is organized in panel formats:

### Regular Price Indices:
- Rows represent cities (70 major cities in China)
- Columns represent months (YYYY-MM format)
- Values are month-over-month percentage changes in housing prices

### Category Indices:
- Rows represent city-housing size combinations (e.g., "北京_90平方米及以下")
- First column is 'housing_size' indicating the size dimension (e.g., "90平方米及以下")
- Remaining columns represent months (YYYY-MM format)
- Values are month-over-month percentage changes in housing prices for specific city-housing size combinations

## Analysis

This data can be used for various analyses:
- Tracking housing price trends across different cities in China
- Comparing price changes between new and second-hand housing markets
- Analyzing differences in price trends by property size categories
- Identifying if certain housing sizes/categories are experiencing different market dynamics
- Examining how different property size segments perform across various cities
- Identifying cities with the fastest growing or declining housing markets in specific size categories 