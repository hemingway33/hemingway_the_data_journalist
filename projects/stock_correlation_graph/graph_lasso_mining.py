#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese Stock Market Correlation Analysis using GraphicalLasso
--------------------------------------------------------------
This script demonstrates how to use the GraphicalLasso estimator to extract
correlations from stock market returns, focusing on Chinese stock market data
from Shanghai and Shenzhen exchanges for the past two years.
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from scipy import stats
import seaborn as sns
import akshare as ak
import networkx as nx
import pickle
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Set font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Create a data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/stock_data')
os.makedirs(data_dir, exist_ok=True)

def fetch_chinese_stock_data(n_stocks=50):
    """
    Fetch stock data for top Chinese stocks from Shanghai and Shenzhen exchanges.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to fetch data for
        
    Returns:
    --------
    quotes : pandas.DataFrame
        DataFrame with stock closing prices
    symbols : list
        List of stock symbols
    names : list
        List of company names
    """
    print("Fetching list of top Chinese stocks...")
    
    # Get top stocks by market cap from Shanghai Exchange (SSE)
    sse_stocks = ak.stock_zh_a_spot_em()
    # Filter for Shanghai stocks (codes start with 6)
    sse_stocks = sse_stocks[sse_stocks['代码'].str.startswith('6')]
    # Sort by market cap (descending)
    sse_stocks = sse_stocks.sort_values(by='总市值', ascending=False)
    
    # Get top stocks by market cap from Shenzhen Exchange (SZSE)
    szse_stocks = ak.stock_zh_a_spot_em()
    # Filter for Shenzhen stocks (codes start with 0 or 3)
    szse_stocks = szse_stocks[szse_stocks['代码'].str.startswith(('0', '3'))]
    # Sort by market cap (descending)
    szse_stocks = szse_stocks.sort_values(by='总市值', ascending=False)
    
    # Combine and take top n_stocks
    top_stocks = pd.concat([sse_stocks.head(n_stocks // 2), szse_stocks.head(n_stocks // 2)])
    top_stocks = top_stocks.sort_values(by='总市值', ascending=False).head(n_stocks)
    
    symbols = top_stocks['代码'].tolist()
    names = top_stocks['名称'].tolist()
    
    # Get end date (today) and start date (2 years ago)
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime('%Y%m%d')
    
    print(f"Downloading stock data for {len(symbols)} stocks...")
    
    # Initialize an empty DataFrame for quotes
    quotes = pd.DataFrame()
    
    # Download data for each stock
    for i, symbol in enumerate(symbols):
        try:
            # Get daily data for the stock
            stock_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                           start_date=start_date, end_date=end_date, 
                                           adjust="qfq")
            
            # Rename the date column and set it as index
            stock_data.rename(columns={'日期': 'Date'}, inplace=True)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)
            
            # Add closing price to the quotes DataFrame
            quotes[symbol] = stock_data['收盘']
            
            print(f"Downloaded data for {names[i]} ({symbol}) - {i+1}/{len(symbols)}")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            # Remove the symbol and name from our lists
            symbols.remove(symbol)
            names.remove(names[i])
    
    # Save the data
    quotes.to_csv(os.path.join(data_dir, 'chinese_stock_prices.csv'))
    
    return quotes, symbols, names

def analyze_stock_correlations(quotes, symbols, names, n_clusters=10):
    """
    Analyze stock correlations using GraphicalLasso.
    
    Parameters:
    -----------
    quotes : pandas.DataFrame
        DataFrame with stock closing prices
    symbols : list
        List of stock symbols
    names : list
        List of company names
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    labels : numpy.ndarray
        Cluster labels for each stock
    embedding : numpy.ndarray
        2D embedding of stocks based on precision matrix
    """
    # Calculate the log returns
    X = quotes.pct_change().dropna()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate empirical covariance and correlation matrices
    emp_cov = X.cov()
    emp_corr = X.corr()
    
    # Use Ledoit-Wolf shrinkage for improved covariance estimation
    lw_cov, _ = ledoit_wolf(X)
    lw_corr = sns.clustermap(lw_cov, vmin=-1, vmax=1, cmap='RdBu_r').data2d
    
    # Estimate the precision matrix with GraphicalLassoCV
    print("Estimating precision matrix with GraphicalLassoCV...")
    model = GraphicalLassoCV(alphas=np.logspace(-3, 0, 10), cv=5, max_iter=200, tol=1e-4, verbose=1)
    model.fit(X)
    
    # Get the precision matrix (inverse covariance)
    precision = model.precision_
    alpha = model.alpha_
    
    print(f"Best alpha: {alpha:.4f}")
    
    # Compute a correlation matrix from the precision matrix
    d = np.sqrt(np.diag(precision))
    precision_corr = precision / np.outer(d, d)
    
    # Cluster the stocks based on the precision correlation matrix
    print("Clustering stocks...")
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed',
                                   assign_labels='discretize', 
                                   random_state=42)
    
    # Transform the precision correlation to a similarity matrix
    similarity = np.abs(precision_corr)
    np.fill_diagonal(similarity, 0)
    
    # Fit the clustering model
    labels = clustering.fit_predict(similarity)
    
    # Create a 2D embedding of stocks for visualization
    embedding = SpectralEmbedding(n_components=2, 
                                 affinity='precomputed',
                                 random_state=42).fit_transform(similarity)
    
    return labels, embedding, precision_corr, emp_corr, alpha

def plot_results(quotes, symbols, names, labels, embedding, precision_corr, emp_corr, alpha, n_clusters):
    """
    Plot the results of stock correlation analysis.
    
    Parameters:
    -----------
    quotes : pandas.DataFrame
        DataFrame with stock closing prices
    symbols : list
        List of stock symbols
    names : list
        List of company names
    labels : numpy.ndarray
        Cluster labels for each stock
    embedding : numpy.ndarray
        2D embedding of stocks based on precision matrix
    precision_corr : numpy.ndarray
        Correlation matrix from precision matrix
    emp_corr : pandas.DataFrame
        Empirical correlation matrix
    alpha : float
        Best alpha parameter from GraphicalLassoCV
    n_clusters : int
        Number of clusters
    """
    # Define cluster colors
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    # Plot 1: Stock prices over time
    ax1 = plt.subplot(gs[0, 0:2])
    quotes_norm = quotes / quotes.iloc[0]  # Normalize to starting price
    for i, symbol in enumerate(symbols):
        ax1.plot(quotes_norm.index, quotes_norm[symbol], 
                alpha=0.7, lw=1, label=f"{symbol}")
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.set_title('Stock Price Evolution (2 years)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    # Plot 2: Graphical Lasso correlation matrix
    ax2 = plt.subplot(gs[0, 2])
    mask = np.zeros_like(precision_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(precision_corr, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1, 
               square=True, linewidths=0, ax=ax2)
    ax2.set_title(f'Sparse Precision Matrix Correlation (alpha={alpha:.4f})')
    
    # Plot 3: Empirical correlation matrix
    ax3 = plt.subplot(gs[1, 0])
    mask = np.zeros_like(emp_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(emp_corr, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1, 
               square=True, linewidths=0, ax=ax3)
    ax3.set_title('Empirical Correlation Matrix')
    
    # Plot 4: Network graph visualization
    ax4 = plt.subplot(gs[1, 1:])
    
    # Plot each stock as a point
    for i in range(len(symbols)):
        ax4.scatter(embedding[i, 0], embedding[i, 1], 
                   color=cluster_colors[labels[i]], 
                   s=100, alpha=0.8)
        # Use only symbol (not Chinese names) to avoid font issues
        ax4.text(embedding[i, 0], embedding[i, 1], symbols[i], 
                fontsize=10, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Draw edges for strongest correlations
    threshold = 0.3  # Only show strong correlations
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            if abs(precision_corr[i, j]) > threshold:
                ax4.plot([embedding[i, 0], embedding[j, 0]], 
                        [embedding[i, 1], embedding[j, 1]], 
                        'k-', alpha=0.3 * abs(precision_corr[i, j]) + 0.1, 
                        linewidth=2 * abs(precision_corr[i, j]))
    
    # Add a legend for clusters
    for i in range(n_clusters):
        ax4.scatter([], [], color=cluster_colors[i], 
                   label=f'Cluster {i+1}')
    
    ax4.legend(loc='upper right')
    ax4.set_title('Stock Correlation Network')
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'chinese_stock_correlations.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, 'chinese_stock_correlations.pdf'), bbox_inches='tight')
    plt.show()

def create_network_graph(symbols, precision_corr, labels, threshold=0.01):
    """
    Create a NetworkX graph from the precision correlation matrix.
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols
    precision_corr : numpy.ndarray
        Correlation matrix from precision matrix
    labels : numpy.ndarray
        Cluster labels for each stock
    threshold : float
        Threshold for including edges (only correlations above this value)
        
    Returns:
    --------
    G : networkx.Graph
        Graph representing the stock correlation network
    """
    # Create an empty undirected graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, symbol in enumerate(symbols):
        G.add_node(symbol, cluster=int(labels[i]))
    
    # Add edges for correlations above threshold
    edge_count = 0
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            corr = abs(precision_corr[i, j])
            if corr > threshold:
                G.add_edge(symbols[i], symbols[j], weight=corr)
                edge_count += 1
    
    print(f"Created network with {len(symbols)} nodes and {edge_count} edges (threshold: {threshold})")
    
    # If very few edges, try with a lower threshold
    if edge_count < len(symbols) // 10:
        print(f"Warning: Very few edges in the network. Consider lowering the threshold below {threshold}")
    
    return G

def save_network_graph(G, output_path):
    """
    Save a NetworkX graph to multiple formats.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to save
    output_path : str
        Base path for saving (without extension)
    """
    # Save as pickle (most complete, preserves all attributes)
    pickle.dump(G, open(f"{output_path}.pickle", 'wb'))
    
    # Save as GraphML (good for importing to other network analysis tools)
    nx.write_graphml(G, f"{output_path}.graphml")
    
    # Save as GML
    nx.write_gml(G, f"{output_path}.gml")
    
    # Save as edgelist (simple format, only edges with weights)
    nx.write_weighted_edgelist(G, f"{output_path}.edgelist")
    
    # Save as GEXF (can be used with Gephi)
    nx.write_gexf(G, f"{output_path}.gexf")
    
    print(f"Network saved in multiple formats at {output_path}.*")

def main():
    """Main function to run the analysis."""
    n_stocks = 1000 # Number of stocks to analyze
    n_clusters = 50  # Number of clusters to form
    
    # Check if we have cached data
    cache_file = os.path.join(data_dir, 'chinese_stock_prices.csv')
    if os.path.exists(cache_file):
        print(f"Loading cached stock data from {cache_file}")
        quotes = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Extract symbols and drop any columns with too many missing values
        min_records = len(quotes) * 0.9  # Require at least 90% of data points
        valid_columns = quotes.columns[quotes.count() >= min_records]
        quotes = quotes[valid_columns]
        symbols = quotes.columns.tolist()
        
        # Get stock names (simplified for cached data)
        names = symbols.copy()
        try:
            # Try to get stock names, fallback to symbols if not possible
            stock_list = ak.stock_zh_a_spot_em()
            name_map = dict(zip(stock_list['代码'], stock_list['名称']))
            names = [name_map.get(symbol, symbol) for symbol in symbols]
        except:
            print("Could not fetch stock names, using symbols instead")
    else:
        # Fetch fresh data
        quotes, symbols, names = fetch_chinese_stock_data(n_stocks)
    
    # Make sure we have enough stocks with data
    if len(symbols) < 10:
        print("Not enough stocks with valid data. Please increase n_stocks or check data source.")
        return
    
    # Analyze stock correlations
    labels, embedding, precision_corr, emp_corr, alpha = analyze_stock_correlations(
        quotes, symbols, names, n_clusters=n_clusters)
    
    # Create and save the network graph
    G = create_network_graph(symbols, precision_corr, labels)
    network_output_path = os.path.join(data_dir, 'chinese_stock_network')
    save_network_graph(G, network_output_path)
    
    # Plot the results
    plot_results(quotes, symbols, names, labels, embedding, precision_corr, 
                emp_corr, alpha, n_clusters)

if __name__ == "__main__":
    main()
