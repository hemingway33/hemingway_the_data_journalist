# Chinese Stock Market Correlation Analysis using GraphicalLasso

This project demonstrates how to use the GraphicalLasso algorithm to analyze and visualize correlations between Chinese stocks from the Shanghai and Shenzhen exchanges.

## Overview

The GraphicalLasso algorithm is used to estimate a sparse precision matrix (inverse covariance) from stock return data. This can reveal underlying patterns of dependencies between stocks while filtering out weak or spurious correlations.

The project includes:
- Automatic downloading of stock data for top Chinese stocks using the akshare library
- Data preprocessing and normalization
- Correlation analysis using GraphicalLasso
- Cluster analysis using Spectral Clustering
- Interactive visualization of stock correlations and clusters
- Export of correlation networks in multiple NetworkX formats for further analysis

## Requirements

- Python 3.7+
- Dependencies: pandas, numpy, matplotlib, seaborn, akshare, scikit-learn, networkx

## Files in this Project

- `graph_lasso_mining.py` - Main script that performs the GraphicalLasso analysis and generates the visualizations
- `analyze_network.py` - Script to analyze the saved network using NetworkX algorithms
- `export_to_gephi.py` - Script to prepare and export the network for visualization in Gephi
- `README.md` - This file

## Usage

To run the main analysis:

```bash
python graph_lasso_mining.py
```

The script will:
1. Download data for top Chinese stocks from Shanghai and Shenzhen exchanges
2. Calculate the correlation and precision matrices
3. Apply spectral clustering to identify groups of related stocks
4. Save the network graph in multiple formats for further analysis
5. Generate visualizations showing:
   - Stock price evolution
   - Sparse precision matrix
   - Empirical correlation matrix
   - Network graph of stock correlations

### Advanced Network Analysis

After running the main script, you can perform additional network analysis:

```bash
python analyze_network.py
```

This will:
1. Load the saved network graph
2. Calculate various network metrics (centrality, clustering, etc.)
3. Generate a network visualization with enhanced attributes
4. Analyze network resilience by simulating node removals

### Exporting to Gephi

To prepare the network for advanced visualization in Gephi:

```bash
python export_to_gephi.py
```

This will:
1. Load the saved network graph
2. Enrich it with additional node and edge attributes
3. Export it in multiple formats suitable for Gephi
4. Provide instructions for using the files in Gephi

## Visualization

The visualization consists of:
- Stock price trends (normalized to starting value)
- Heatmap of the sparse precision matrix (showing direct influences)
- Heatmap of the empirical correlation matrix
- Network graph where:
  - Nodes represent stocks, colored by cluster
  - Edges represent strong correlations
  - Node size represents market capitalization

The results are saved as PNG and PDF files in the `data/stock_data/` directory.

## Network Graph Formats

The correlation network is saved in multiple formats for further analysis:

1. **Pickle (.pickle)** - Complete Python object serialization, preserves all graph attributes
   - Use with: `G = pickle.load(open('chinese_stock_network.pickle', 'rb'))`

2. **GraphML (.graphml)** - XML-based format compatible with many network analysis tools
   - Use with: `G = nx.read_graphml('chinese_stock_network.graphml')`
   - Compatible with: NetworkX, igraph, Gephi, Cytoscape

3. **GML (.gml)** - Graph Modeling Language format
   - Use with: `G = nx.read_gml('chinese_stock_network.gml')`
   - Compatible with: NetworkX, Gephi, SNAP

4. **Edge List (.edgelist)** - Simple text format listing edges and weights
   - Use with: `G = nx.read_weighted_edgelist('chinese_stock_network.edgelist')`
   - Compatible with most network analysis tools

5. **GEXF (.gexf)** - XML-based format specifically designed for Gephi
   - Use with: `G = nx.read_gexf('chinese_stock_network.gexf')`
   - Best for visualization in Gephi

All network files are saved in the `data/stock_data/` directory.

## Network Analysis

The saved network can be loaded and analyzed using NetworkX or other network analysis tools:

```python
import networkx as nx
import pickle

# Load the network from pickle (most complete format)
G = pickle.load(open('data/stock_data/chinese_stock_network.pickle', 'rb'))

# Basic network statistics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average clustering coefficient: {nx.average_clustering(G)}")
print(f"Network density: {nx.density(G)}")

# Community detection
from networkx.algorithms import community
communities = community.greedy_modularity_communities(G)
print(f"Number of communities: {len(communities)}")

# Centrality measures
centrality = nx.betweenness_centrality(G)
most_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Most central stocks:", most_central)
```

## Methodology

The analysis follows these steps:
1. Data fetching and preprocessing
2. Calculation of stock returns
3. Estimating the precision matrix using GraphicalLassoCV (with cross-validation)
4. Clustering stocks based on this precision matrix
5. Creating a 2D embedding for visualization
6. Creating and saving a NetworkX graph representation
7. Generating the final visualization

## References

This implementation is inspired by the scikit-learn example:
https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html

## License

See the main project license file. 