# Stats.gov.cn Category Visualization

This directory contains D3.js visualizations for the National Bureau of Statistics of China article categories.

## Files

- `stats_gov_cn_category_chart.html` - Interactive D3.js donut chart visualization
- `stats_gov_cn_category_chart_static.html` - Static SVG version (fallback version)

## Running the Visualization

Due to browser security restrictions (CORS policy), you'll need to run a local web server to properly view these visualizations. Here's how:

### Step 1: Open a Terminal

Open a terminal or command prompt.

### Step 2: Navigate to Project Root

Navigate to the root directory of the project:

```bash
cd /path/to/hemingway_the_data_journalist
```

### Step 3: Start a Local Server

Run a simple Python HTTP server:

```bash
python -m http.server 8000
```

### Step 4: View the Visualization

Open your web browser and go to:

- Interactive version:
  ```
  http://localhost:8000/tools/interactive_plot/stats_gov_cn_category_chart.html
  ```

- Static version (fallback):
  ```
  http://localhost:8000/tools/interactive_plot/stats_gov_cn_category_chart_static.html
  ```

### Step 5: Stop the Server

When you're done, go back to the terminal and press `Ctrl+C` to stop the server.

## Troubleshooting

If you see only the title without the chart in the interactive version:

1. Check browser's developer console (F12) for errors
2. Try the static version instead, which doesn't require loading external data
3. Verify the data file exists at: `data/stats_gov_cn/stats_gov_cn_articles_20250329_144754.json` 