import graphviz

def plot_tree(ds, node, parent_name, graph, counter, decision=None):
    # Base case: leaf node
    if node.is_leaf():
        leaf_name = f'leaf_{counter}'
        graph.node(leaf_name, label=str(ds.y_decoder[node.value]), shape='box')
        graph.edge(parent_name, leaf_name, label=decision)
        return counter + 1

    # Displaying Case Analysis in the current node
    feature = str(ds.cols[node.feature])
    if ('_' in feature):
        feat, value = feature.split('_')
        if (feat in ds.cat_cols):
            attribute_name = f"{feat} == {value} ?"
        else:
            attribute_name = f"{feature} <= {node.threshold}"
    else:
        attribute_name = f"{feature} <= {node.threshold}"
    
    # Adding the Labels to the nodes and braches
    internal_name = f'internal_{counter}'
    graph.node(internal_name, label=attribute_name)
    if parent_name is not None:
        graph.edge(parent_name, internal_name, label=decision)

    # Recursive calls
    # Left child
    counter = plot_tree(ds, node.left, internal_name, graph, counter + 1, "No")
    # Right child
    counter = plot_tree(ds, node.right, internal_name, graph, counter, "Yes")
    
    return counter

def visualize_tree(decision_tree, dataset, file_path=None):
    graph = graphviz.Digraph(format='png', node_attr={'color': 'lightblue2', 'style': 'filled'})
    plot_tree(dataset, decision_tree.root, None, graph, 0)
    if (file_path is not None):
        graph.render(file_path)
    return graph