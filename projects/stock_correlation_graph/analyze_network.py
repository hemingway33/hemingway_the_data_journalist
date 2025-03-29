#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Correlation Network Analysis
---------------------------------
This script demonstrates how to load and analyze the saved stock correlation network
created by the graph_lasso_mining.py script.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community, centrality

# Create a function to load the graph from various formats
def load_graph(path, format='pickle'):
    """
    Load a NetworkX graph from a file.
    
    Parameters:
    -----------
    path : str
        Path to the graph file without extension
    format : str
        Format to load (pickle, graphml, gml, edgelist, gexf)
    
    Returns:
    --------
    G : networkx.Graph
        Loaded graph
    """
    if format == 'pickle':
        return pickle.load(open(f"{path}.pickle", 'rb'))
    elif format == 'graphml':
        return nx.read_graphml(f"{path}.graphml")
    elif format == 'gml':
        return nx.read_gml(f"{path}.gml")
    elif format == 'edgelist':
        return nx.read_weighted_edgelist(f"{path}.edgelist")
    elif format == 'gexf':
        return nx.read_gexf(f"{path}.gexf")
    else:
        raise ValueError(f"Unknown format: {format}")

def analyze_network(G):
    """
    Analyze a stock correlation network.
    
    Parameters:
    -----------
    G : networkx.Graph
        Stock correlation network
    """
    # Basic network statistics
    print("\n=== Basic Network Statistics ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.4f}")
    
    # Check if the graph is connected
    if nx.is_connected(G):
        print("The network is connected.")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.4f}")
    else:
        print("The network is not connected.")
        components = list(nx.connected_components(G))
        print(f"Number of connected components: {len(components)}")
        largest_component = max(components, key=len)
        print(f"Size of largest component: {len(largest_component)} nodes")
    
    # Clustering coefficient
    avg_clustering = nx.average_clustering(G, weight='weight')
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    
    # Community detection
    print("\n=== Community Analysis ===")
    communities = list(community.greedy_modularity_communities(G, weight='weight'))
    print(f"Number of communities detected: {len(communities)}")
    community_sizes = [len(c) for c in communities]
    print(f"Sizes of top 5 communities: {sorted(community_sizes, reverse=True)[:5]}")
    
    # Centrality measures
    print("\n=== Centrality Analysis ===")
    
    # Degree centrality
    degree_cent = centrality.degree_centrality(G)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 stocks by degree centrality:")
    for stock, value in top_degree:
        print(f"  {stock}: {value:.4f}")
    
    # Betweenness centrality
    print("\nCalculating betweenness centrality (this may take a while)...")
    betweenness_cent = centrality.betweenness_centrality(G, weight='weight')
    top_betweenness = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 stocks by betweenness centrality:")
    for stock, value in top_betweenness:
        print(f"  {stock}: {value:.4f}")
    
    # Eigenvector centrality
    try:
        eigenvector_cent = centrality.eigenvector_centrality_numpy(G, weight='weight')
        print("\nTop 10 stocks by eigenvector centrality:")
        top_eigenvector = sorted(eigenvector_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        for stock, value in top_eigenvector:
            print(f"  {stock}: {value:.4f}")
    except nx.AmbiguousSolution:
        print("\nCould not calculate eigenvector centrality for the whole graph (disconnected).")
        print("Calculating for the largest connected component instead...")
        
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        
        if len(G_cc) > 1:
            eigenvector_cent_cc = centrality.eigenvector_centrality_numpy(G_cc, weight='weight')
            
            # Create a dictionary for all nodes (with 0 for nodes not in largest component)
            eigenvector_cent = {node: 0.0 for node in G.nodes()}
            eigenvector_cent.update(eigenvector_cent_cc)
            
            print(f"Eigenvector centrality calculated for largest component ({len(G_cc)} nodes)")
            top_eigenvector = sorted(eigenvector_cent_cc.items(), key=lambda x: x[1], reverse=True)[:10]
            print("Top 10 stocks by eigenvector centrality (in largest component):")
            for stock, value in top_eigenvector:
                print(f"  {stock}: {value:.4f}")
        else:
            print("Largest component too small for eigenvector centrality.")
            eigenvector_cent = {node: 0.0 for node in G.nodes()}
    except Exception as e:
        print(f"\nError calculating eigenvector centrality: {e}")
        print("Using degree centrality as a substitute.")
        # Normalize degree centrality to [0,1] range
        max_degree = max(degree_cent.values()) if degree_cent.values() else 1
        eigenvector_cent = {n: v/max_degree for n, v in degree_cent.items()}
    
    return {
        'degree_centrality': degree_cent,
        'betweenness_centrality': betweenness_cent,
        'eigenvector_centrality': eigenvector_cent,
        'communities': communities
    }

def visualize_network(G, metrics, output_path=None):
    """
    Visualize the network with metrics.
    
    Parameters:
    -----------
    G : networkx.Graph
        Stock correlation network
    metrics : dict
        Dictionary of network metrics
    output_path : str, optional
        Path to save the visualization
    """
    # Check if the graph is too large or disconnected
    if G.number_of_nodes() > 100 or not nx.is_connected(G):
        print("\nGraph is large or disconnected. Visualizing only the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        G_vis = G.subgraph(largest_cc).copy()
        print(f"Largest component has {G_vis.number_of_nodes()} nodes and {G_vis.number_of_edges()} edges.")
        
        if G_vis.number_of_nodes() < 5:
            print("Largest component is too small for meaningful visualization.")
            return
    else:
        G_vis = G
    
    plt.figure(figsize=(12, 12))
    
    # Assign community as a node attribute
    community_map = {}
    for i, comm in enumerate(metrics['communities']):
        for node in comm:
            community_map[node] = i
    
    # Create position using spring layout
    print("Computing layout...")
    pos = nx.spring_layout(G_vis, seed=42, k=0.3, iterations=50)
    
    # Get nodes in the visualization graph
    nodes_in_vis = list(G_vis.nodes())
    
    # Get node sizes based on degree centrality for the visualization subgraph
    node_size = []
    node_color = []
    for n in nodes_in_vis:
        # Use either eigenvector centrality or degree as fallback
        size_value = metrics['degree_centrality'][n] * 3000
        node_size.append(size_value)
        
        # Use community as color if the node is in a community
        if n in community_map:
            node_color.append(community_map[n])
        else:
            node_color.append(0)
    
    # Draw the network
    print("Drawing network...")
    edges = nx.draw_networkx_edges(
        G_vis, pos, 
        alpha=0.2, 
        width=[G_vis[u][v].get('weight', 1) * 2 for u, v in G_vis.edges()]
    )
    nodes = nx.draw_networkx_nodes(
        G_vis, pos, 
        node_size=node_size,
        node_color=node_color, 
        cmap=plt.cm.tab20,
        alpha=0.8
    )
    
    # Add labels to the most central nodes
    # Get top nodes by degree centrality in the visualization subgraph
    degree_in_vis = {n: metrics['degree_centrality'][n] for n in nodes_in_vis}
    top_degree = sorted(degree_in_vis.items(), key=lambda x: x[1], reverse=True)[:min(20, len(nodes_in_vis))]
    labels = {node: node for node, _ in top_degree}
    nx.draw_networkx_labels(G_vis, pos, labels=labels, font_size=8, font_weight='bold')
    
    plt.title('Stock Correlation Network (Largest Connected Component)')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main function to run the analysis."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Calculate path to the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'data/stock_data')
    network_path = os.path.join(data_dir, 'chinese_stock_network')
    
    try:
        # Try to load from pickle (most complete)
        print(f"Loading network from {network_path}.pickle")
        G = load_graph(network_path, format='pickle')
    except FileNotFoundError:
        print(f"Could not find {network_path}.pickle")
        print("Please run graph_lasso_mining.py first to generate the network.")
        return
    
    # Analyze the network
    metrics = analyze_network(G)
    
    # Visualize the network
    output_path = os.path.join(data_dir, 'network_analysis.png')
    visualize_network(G, metrics, output_path)
    
    # Additional analysis: network resilience
    print("\n=== Network Resilience Analysis ===")
    
    # Original network properties
    original_largest_cc = len(max(nx.connected_components(G), key=len))
    
    # Remove nodes by degree centrality
    resilience_data = []
    G_temp = G.copy()
    nodes_by_centrality = sorted(metrics['degree_centrality'].items(), key=lambda x: x[1], reverse=True)
    
    for i, (node, _) in enumerate(nodes_by_centrality[:30]):  # Remove top 30 nodes
        G_temp.remove_node(node)
        largest_cc = len(max(nx.connected_components(G_temp), key=len))
        resilience_data.append((i+1, largest_cc / original_largest_cc))
    
    # Plot resilience
    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in resilience_data], [x[1] for x in resilience_data], 'o-', linewidth=2)
    plt.xlabel('Number of Central Nodes Removed')
    plt.ylabel('Fraction of Largest Connected Component Size')
    plt.title('Network Resilience to Targeted Node Removal')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(data_dir, 'network_resilience.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 