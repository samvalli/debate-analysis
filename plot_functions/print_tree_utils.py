from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def print_tree(data, key,score, title):
    nodes = {}

    # Create the root node '0' with a placeholder z-score if needed
    nodes['0'] = Node(title, z_score=None)

    # Build the tree by creating nodes and setting their parents
    for _, row in data.iterrows():
        parent_id = row['parent_id']
        child_id = row['id']
        z_score = row[score]  # Assuming 'z-score' is a column in your DataFrame
        
        # Create child node with z_score and set parent
        nodes[child_id] = Node(child_id, parent=nodes[parent_id], z_score=z_score)

    # Print the tree structure in text, showing z-score instead of the ID
    for pre, _, node in RenderTree(nodes['0']):
        # Print z-score, if available, else print the node name
        print(f"{pre}{node.z_score if node.z_score is not None else node.name}")

    # Optional: Export the tree to a .dot file and view as an image if Graphviz is available
    #DotExporter(nodes['0']).to_dotfile(title+'_'+key+'.dot')
    return 

def print_tree_score(data, key,score, title):
    nodes = {}

    # Create the root node '0' with a placeholder z-score if needed
    nodes['0'] = Node(title, z_score=None)

    # Build the tree by creating nodes and setting their parents
    for _, row in data.iterrows():
        parent_id = row['parent_id']
        child_id = row['id']
        z_score = row[score]  # Assuming 'z-score' is a column in your DataFrame
        
        # Create child node with z_score and set parent
        nodes[child_id] = Node(child_id, parent=nodes[parent_id], z_score=z_score)

    # Print the tree structure in text, showing z-score instead of the ID
    for pre, _, node in RenderTree(nodes['0']):
        # Print z-score, if available, else print the node name
        print(f"{pre}{node.z_score if node.z_score is not None else node.name}")

    # Optional: Export the tree to a .dot file and view as an image if Graphviz is available
    #DotExporter(nodes['0']).to_dotfile(title+'_'+key+'.dot')
    return 