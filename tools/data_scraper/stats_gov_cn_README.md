# Data Scraper Tools

## Stats.gov.cn Scraper

This scraper collects article listings from the National Bureau of Statistics of China's website (stats.gov.cn).

### Features

- Collects article listings page by page from the statistical information release section
- Extracts title, URL, and publication date for each article
- Automatically categorizes articles based on their content
- Saves data in both JSON and CSV formats
- Provides statistics about the collected data

### Usage

```bash
python stats_gov_cn.py [options]
```

#### Options:

- `--max-pages NUM`: Maximum number of pages to scrape (default: all available)
- `--output-dir DIR`: Directory to store output files (default: data/stats_gov_cn)
- `--delay SECONDS`: Delay between requests in seconds (default: 1.0)
- `--quiet`: Suppress detailed output

#### Examples:

Scrape all available pages:
```bash
python stats_gov_cn.py
```

Scrape only the first 5 pages:
```bash
python stats_gov_cn.py --max-pages 5
```

Scrape with a custom output directory:
```bash
python stats_gov_cn.py --output-dir data/custom_output
```

Scrape with a faster request rate (0.5 second delay):
```bash
python stats_gov_cn.py --delay 0.5
```

### Output

The scraper generates two output files with timestamps in the filename:

1. `stats_gov_cn_articles_YYYYMMDD_HHMMSS.json`: JSON file containing all collected articles
2. `stats_gov_cn_articles_YYYYMMDD_HHMMSS.csv`: CSV file containing all collected articles

Each article entry contains:
- Title: The title of the article
- URL: The full URL to the article
- Date: The publication date (YYYY-MM-DD format)
- Category: Automatically assigned category based on content
- Source: "National Bureau of Statistics of China"

### Article Categories

Articles are automatically categorized into the following categories:
- GDP: Articles about GDP and economic growth
- CPI: Articles about Consumer Price Index and inflation
- PPI: Articles about Producer Price Index
- PMI: Articles about Purchasing Managers' Index
- Industrial Production: Articles about industrial production and enterprise performance
- Fixed Asset Investment: Articles about fixed asset investments
- Retail Sales: Articles about retail sales and consumption
- Real Estate: Articles about real estate and housing
- Trade: Articles about imports, exports and foreign trade
- Employment: Articles about employment and unemployment rates
- Energy: Articles about energy production and consumption
- Population: Articles about population and demographics
- Other: Articles that don't fit into the above categories 