import networkx as nx
import numpy as np
from collections import Counter
import pandas as pd


def find_families(components, node_family_dict, pass_nans=True):
    component_classes = []
    component_nodes = []

    for component in components:
        families = [node_family_dict[node] for node in component if
                    node in node_family_dict and not pd.isna(node_family_dict[node])]
        if families and pass_nans:
            most_common_class = Counter(families).most_common(1)[0][0]
            component_classes.append(most_common_class)
            component_nodes.append(component)
        else:
            component_classes.append(np.nan)
            component_nodes.append(component)
    return component_nodes, component_classes