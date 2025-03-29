#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting Stock Correlation Network for Gephi
--------------------------------------------
This script demonstrates how to prepare and export the stock correlation network
for visualization in Gephi, a popular network visualization tool.
"""

import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import akshare as ak

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

def enrich_graph_for_gephi(G):
    """
    Enrich the graph with additional attributes for better Gephi visualization.
    
    Parameters:
    -----------
    G : networkx.Graph
        Stock correlation network
    
    Returns:
    --------
    G : networkx.Graph
        Enriched graph
    """
    # Make a copy to avoid modifying the original
    G_enriched = G.copy()
    
    # Try to get stock details for better labeling
    try:
        # Get all Chinese A stocks from akshare
        stock_list = ak.stock_zh_a_spot_em()
        
        # Print columns for debugging
        print("Available columns in stock_list:")
        print(stock_list.columns.tolist())
        
        # Create mappings from stock code to other attributes
        name_map = dict(zip(stock_list['代码'], stock_list['名称']))
        
        # Get market cap if available
        if '总市值' in stock_list.columns:
            marketcap_map = dict(zip(stock_list['代码'], stock_list['总市值']))
        else:
            marketcap_map = {}
            print("Market cap column not available")
        
        # Add node attributes
        for node in G_enriched.nodes():
            # Add name attribute
            if node in name_map:
                G_enriched.nodes[node]['name'] = name_map[node]
            else:
                G_enriched.nodes[node]['name'] = node
                
            # Add cluster attribute
            G_enriched.nodes[node]['cluster'] = G_enriched.nodes[node].get('cluster', 0)
                
            # Add market cap attribute (convert to float for easy filtering in Gephi)
            if node in marketcap_map:
                try:
                    G_enriched.nodes[node]['marketcap'] = float(marketcap_map[node])
                except (ValueError, TypeError):
                    G_enriched.nodes[node]['marketcap'] = 0.0
            else:
                G_enriched.nodes[node]['marketcap'] = 0.0
        
        print("Successfully enriched graph with stock details.")
    except Exception as e:
        print(f"Could not enrich graph with stock details: {e}")
        print("Continuing with basic node attributes.")
    
    # Calculate network metrics to add as node attributes
    print("Calculating network metrics...")
    
    # Degree
    degree = dict(G_enriched.degree())
    nx.set_node_attributes(G_enriched, degree, 'degree')
    
    # Eigenvector centrality
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G_enriched, weight='weight')
        nx.set_node_attributes(G_enriched, eigenvector, 'eigenvector_centrality')
    except:
        print("Could not calculate eigenvector centrality. Using degree instead.")
        nx.set_node_attributes(G_enriched, degree, 'eigenvector_centrality')
    
    # Betweenness centrality (may be slow for large networks)
    if G_enriched.number_of_nodes() < 1000:
        try:
            betweenness = nx.betweenness_centrality(G_enriched, weight='weight')
            nx.set_node_attributes(G_enriched, betweenness, 'betweenness_centrality')
        except:
            print("Could not calculate betweenness centrality.")
    else:
        print("Network too large for betweenness centrality calculation.")
    
    # Calculate clustering coefficient
    clustering = nx.clustering(G_enriched, weight='weight')
    nx.set_node_attributes(G_enriched, clustering, 'clustering')
    
    return G_enriched

def export_for_gephi(G, output_dir):
    """
    Export the graph in multiple formats suitable for Gephi.
    
    Parameters:
    -----------
    G : networkx.Graph
        Stock correlation network with enriched attributes
    output_dir : str
        Directory to save the exported files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. GEXF format (best for Gephi)
    gexf_path = os.path.join(output_dir, 'stock_network_enriched.gexf')
    nx.write_gexf(G, gexf_path)
    print(f"Exported GEXF to {gexf_path}")
    
    # 2. GraphML format
    graphml_path = os.path.join(output_dir, 'stock_network_enriched.graphml')
    nx.write_graphml(G, graphml_path)
    print(f"Exported GraphML to {graphml_path}")
    
    # 3. Export node attributes to CSV for further analysis
    node_data = []
    for node, attrs in G.nodes(data=True):
        node_dict = {'id': node}
        node_dict.update(attrs)
        node_data.append(node_dict)
    
    node_df = pd.DataFrame(node_data)
    csv_path = os.path.join(output_dir, 'stock_node_attributes.csv')
    node_df.to_csv(csv_path, index=False)
    print(f"Exported node attributes to {csv_path}")
    
    # 4. Export edge list to CSV
    edge_data = []
    for u, v, attrs in G.edges(data=True):
        edge_dict = {'source': u, 'target': v}
        edge_dict.update(attrs)
        edge_data.append(edge_dict)
    
    edge_df = pd.DataFrame(edge_data)
    edge_csv_path = os.path.join(output_dir, 'stock_edge_list.csv')
    edge_df.to_csv(edge_csv_path, index=False)
    print(f"Exported edge list to {edge_csv_path}")

def print_gephi_instructions():
    """Print instructions for using the exported files in Gephi."""
    print("\n=== How to Use These Files in Gephi ===")
    print("1. Download Gephi from https://gephi.org/")
    print("2. Open Gephi and create a new project")
    print("3. Import the .gexf file (File > Open)")
    print("4. In the 'Overview' tab:")
    print("   - Run layout algorithms (e.g., ForceAtlas2)")
    print("   - Adjust node sizes based on attributes (e.g., 'eigenvector_centrality' or 'marketcap')")
    print("   - Color nodes by attributes (e.g., 'industry' or 'cluster')")
    print("5. Run the Modularity community detection algorithm:")
    print("   - Statistics > Network Overview > Modularity")
    print("   - Then color nodes by the 'modularity_class' attribute")
    print("6. Explore the network using Gephi's tools:")
    print("   - Filter nodes based on attributes")
    print("   - Highlight node neighborhoods")
    print("   - Adjust edge weights and opacity")
    print("7. In the 'Preview' tab, customize the final visualization")
    print("8. Export the visualization (File > Export > SVG/PDF/PNG)")
    print("\nFor detailed Gephi tutorials, visit: https://gephi.org/users/")

def main():
    """Main function to run the export process."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Calculate path to the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'data/stock_data')
    network_path = os.path.join(data_dir, 'chinese_stock_network')
    gephi_output_dir = os.path.join(data_dir, 'gephi_export')
    
    try:
        # Try to load from pickle (most complete)
        print(f"Loading network from {network_path}.pickle")
        G = load_graph(network_path, format='pickle')
    except FileNotFoundError:
        print(f"Could not find {network_path}.pickle")
        print("Please run graph_lasso_mining.py first to generate the network.")
        return
    
    # Enrich the graph with additional attributes for Gephi
    G_enriched = enrich_graph_for_gephi(G)
    
    # Export the enriched graph for Gephi
    export_for_gephi(G_enriched, gephi_output_dir)
    
    # Print instructions for using in Gephi
    print_gephi_instructions()

if __name__ == "__main__":
    main() 