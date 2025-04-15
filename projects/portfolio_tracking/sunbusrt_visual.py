import pandas as pd
import plotly.graph_objects as go
import json

# Read the data
try:
    df = pd.read_csv("projects/portfolio_tracking/qqld_periods_cut.csv")
except FileNotFoundError:
    print("Error: projects/portfolio_tracking/qqld_periods_cut.csv not found.")
    exit()

# Ensure 'values' column is numeric
df['values'] = pd.to_numeric(df['values'], errors='coerce')
df = df.dropna(subset=['values']) # Drop rows where conversion failed

# Calculate original total sum and sums per period/general_status for text display
original_total_sum = df['values'].sum()
original_period_sums = df.groupby('period')['values'].sum().to_dict()
original_general_sums = df.groupby(['period', 'general_status'])['values'].sum().unstack(fill_value=0).to_dict('index') # Group by period then general_status

# Dictionary to store node data: nodes[id] = {'label': label, 'parent': parent, 'value': modified_value, 'original_value': original_leaf_value}
nodes = {}

# Process the dataframe to build the node hierarchy
MIN_SLICE_VALUE = 0.5 # Add a small constant to make tiny slices more visible
OVERDUE_MULTIPLIER = 5 # Factor to visually enlarge '逾期' sections

for index, row in df.iterrows():
    period = str(row['period'])
    general = str(row['general_status'])
    specific = str(row['specific_status'])
    original_row_value = float(row['values']) # Store original value from row

    # --- Visual Enhancement for Overdue ---
    inflated_value = original_row_value # Start with original value
    if general == '逾期':
        inflated_value *= OVERDUE_MULTIPLIER
        # print(f"Note: Inflating value for {period} - {general} - {specific} by {OVERDUE_MULTIPLIER}x for visual emphasis.")
    # --- End Visual Enhancement ---

    # Define IDs
    id0 = period
    id1 = f"{period} - {general}"
    id2 = f"{period} - {general} - {specific}"

    # Add/Update Level 0 (Period)
    if id0 not in nodes:
        nodes[id0] = {'label': period, 'parent': "", 'value': 0, 'original_value': 0}

    # Add/Update Level 1 (General Status)
    if id1 not in nodes:
        nodes[id1] = {'label': general, 'parent': id0, 'value': 0, 'original_value': 0}

    # Assign the value ONLY to the appropriate leaf node for this row
    if general == specific:
        # Level 1 node is the leaf
        nodes[id1]['value'] += inflated_value + MIN_SLICE_VALUE
        nodes[id1]['original_value'] += original_row_value # Add original value
    else:
        # Level 2 node is the leaf
        if id2 not in nodes:
            nodes[id2] = {'label': specific, 'parent': id1, 'value': 0, 'original_value': 0}
        nodes[id2]['value'] += inflated_value + MIN_SLICE_VALUE
        nodes[id2]['original_value'] += original_row_value # Add original value


# Convert the nodes dictionary into lists for Plotly
ids = list(nodes.keys())
labels = [nodes[id]['label'] for id in ids]
parents = [nodes[id]['parent'] for id in ids]
values = [nodes[id]['value'] for id in ids] # Modified values for sizing
original_leaf_values_list = [nodes[id]['original_value'] for id in ids] # Original values stored on LEAF nodes

# --- Create Custom Text Labels ---
custom_texts = []
node_colors = []
node_color_map = {}

# Define color mapping using a Blue-centric theme
# Lighter theme for the 'Before Aug' branch
STATUS_COLORS_BEFORE = {
    "结清": "#BBDEFB",  # Light Blue (Material Blue 100)
    "正常": "#BBDEFB",  # Light Blue
    "逾期": "#FFE0B2",  # Light Orange (Material Orange 100)
    "不良": "#FFCDD2",  # Light Red (Material Red 100)
    "DEFAULT": "#ECEFF1" # Light Blue Grey (Material Blue Grey 50)
}
ROOT_COLOR_BEFORE = "#CFD8DC" # Medium-Light Blue Grey (Material Blue Grey 100)

# Darker/Standard theme for the 'After Aug' branch
STATUS_COLORS_AFTER = {
    "结清": "#2196F3",  # Standard Blue (Material Blue 500)
    "正常": "#2196F3",  # Standard Blue
    "逾期": "#FF9800",  # Standard Orange (Material Orange 500)
    "不良": "#F44336",  # Standard Red (Material Red 500)
    "DEFAULT": "#90A4AE" # Standard Blue Grey (Material Blue Grey 300)
}
ROOT_COLOR_AFTER = "#1976D2" # Darker Blue (Material Blue 700)

for i, id_val in enumerate(ids):
    label = labels[i]
    parent_id = parents[i]
    original_node_leaf_value = original_leaf_values_list[i]
    current_color = "#FFFFFF" # Initialize

    # Determine Branch and Assign Colors
    is_after_branch = id_val.startswith("24年8月后(含)投放")
    status_colors_to_use = STATUS_COLORS_AFTER if is_after_branch else STATUS_COLORS_BEFORE
    root_color_to_use = ROOT_COLOR_AFTER if label == "24年8月后(含)投放" else ROOT_COLOR_BEFORE
    default_color = status_colors_to_use["DEFAULT"]

    # --- Text Label Logic (with percentages relative to parent) ---
    text_label = label # Default text

    if parent_id == "": # Innermost circle (Period)
        original_agg_value = original_period_sums.get(label, 0)
        percentage = (original_agg_value / original_total_sum * 100) if original_total_sum else 0
        text_label = f"{label}<br>{original_agg_value:.2f}<br>({percentage:.1f}%)"
        current_color = root_color_to_use

    elif parent_id in original_period_sums: # Second circle (General Status - parent is a Period)
        period_label = parent_id
        general_status_label = label
        own_agg_value = original_general_sums.get(period_label, {}).get(general_status_label, 0)
        parent_agg_value = original_period_sums.get(period_label, 0)
        percentage = (own_agg_value / parent_agg_value * 100) if parent_agg_value else 0
        text_label = f"{label}<br>{own_agg_value:.2f}<br>({percentage:.1f}%)"
        current_color = status_colors_to_use.get(label, default_color)

    else: # Outermost circle (Specific Status - parent is General Status)
        # Need parent's period and general status to get parent's aggregate value
        parent_parts = parent_id.split(' - ', 1) # Split only once
        parent_period_label = parent_parts[0]
        parent_general_status_label = parent_parts[1] if len(parent_parts) > 1 else ""

        own_leaf_value = original_node_leaf_value
        parent_agg_value = original_general_sums.get(parent_period_label, {}).get(parent_general_status_label, 0)
        percentage = (own_leaf_value / parent_agg_value * 100) if parent_agg_value else 0
        text_label = f"{label}<br>{own_leaf_value:.2f}<br>({percentage:.1f}%)"
        parent_color = node_color_map.get(parent_id, default_color)
        current_color = parent_color
    # --- End Text Label Logic ---

    custom_texts.append(text_label) # Add the generated text label
    node_colors.append(current_color)
    node_color_map[id_val] = current_color
# --- End Custom Text Labels & Colors ---

# --- More Debugging ---
print("--- Final Nodes Dictionary ---")
print(json.dumps(nodes, indent=2, ensure_ascii=False))
print("----------------------------")
# --- End Debugging ---

# --- Debugging --- 
print(f"Debugging Info:")
print(f"  Number of IDs: {len(ids)}")
print(f"  Number of Labels: {len(labels)}")
print(f"  Number of Parents: {len(parents)}")
print(f"  Number of Values: {len(values)}")
if ids:
    print(f"  Sample ID: {ids[0]}")
    print(f"  Sample Label: {labels[0]}")
    print(f"  Sample Parent: {parents[0]}")
    print(f"  Sample Value: {values[0]}")
    print(f"  Sum of Values: {sum(v for v in values if isinstance(v, (int, float)))}") # Ensure we sum numbers
# --- End Debugging ---

# Create the figure
fig = go.Figure(go.Sunburst(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values, # Use modified values for sizing
    branchvalues="remainder",
    hoverinfo="label+percent parent+value",
    text=custom_texts,
    textinfo="text",
    marker_colors=node_colors # Assign colors to markers
))

# Update layout for better appearance
fig.update_layout(
    margin=dict(t=50, l=50, r=50, b=50), # Increased margins
    title=dict(text='Portfolio Status Sunburst Chart', x=0.5)
)
fig.show()