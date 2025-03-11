import re
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random

class BayesianNetwork:
    def __init__(self, network_str: str):
        # Parse the network string in the format "[Node][Child|Parent1:Parent2]..."
        self.graph = nx.DiGraph()
        # Find all tokens enclosed in square brackets
        tokens = re.findall(r'\[([^\[\]]+)\]', network_str)
        for token in tokens:
            # If token contains a "|", it has parents
            if "|" in token:
                node, parents_str = token.split("|")
                node = node.strip()
                self.graph.add_node(node)
                parents = parents_str.split(":")
                for parent in parents:
                    parent = parent.strip()
                    self.graph.add_node(parent)
                    # Add edge from parent to node
                    self.graph.add_edge(parent, node)
            else:
                # Token with no parents
                node = token.strip()
                self.graph.add_node(node)

    def plot(self):
        # Plot the Bayesian Network using networkx and matplotlib
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, arrows=True)
        plt.show()

def dsep(bn: BayesianNetwork, x: str, y: str, z: str) -> bool:
    # dsep returns True if x and y are d-separated given z, False otherwise.
    # We implement d-separation test via moralization of the ancestral graph.
    Z = {z}
    # Set of nodes of interest: x, y and the conditioning set Z.
    nodes_of_interest = {x, y} | Z
    # Compute ancestors for all nodes in nodes_of_interest
    A = set(nodes_of_interest)
    for node in list(nodes_of_interest):
        A |= nx.ancestors(bn.graph, node)
    # Induced subgraph over the ancestors and nodes of interest
    induced = bn.graph.subgraph(A).copy()
    # Create an undirected graph from the induced subgraph
    G = nx.Graph()
    G.add_nodes_from(induced.nodes())
    for u, v in induced.edges():
        G.add_edge(u, v)
    # Moralize the graph: for every node, connect all its parents.
    for node in induced.nodes():
        # Get all parents of node that are in A
        parents = [p for p in bn.graph.predecessors(node) if p in A]
        for parent1, parent2 in combinations(parents, 2):
            G.add_edge(parent1, parent2)
    # Remove the conditioned nodes Z from the moral graph
    G.remove_nodes_from(Z)
    # Check if there is a path between x and y in the resulting graph
    if nx.has_path(G, x, y):
        # If there is a path then x and y are d-connected given Z
        return False
    else:
        # Otherwise, x and y are d-separated given Z
        return True

# Create the Bayesian Network using the provided network string
bn = BayesianNetwork("[Z1][Z4|Z2:Z3][Z5|Z3][Z3|Z1][Z2|Z1][Z6|Z4]")
bn.plot()

# Define the number of nodes and create a list of node names
n = 6
nodes = ["Z" + str(i) for i in range(1, n+1)]

# First d-separation test with n3 = [Z6, Z2, Z4]
n3 = [nodes[5], nodes[1], nodes[3]]
ds = dsep(bn, n3[0], n3[1], n3[2])
if ds:
    print(f"{n3[0]} is d-separated from {n3[1]} | {n3[2]}")
else:
    print(f"{n3[0]} is d-connected to {n3[1]} | {n3[2]}")

# Second d-separation test with n3 = [Z2, Z3, Z4]
n3 = [nodes[1], nodes[2], nodes[3]]
ds = dsep(bn, n3[0], n3[1], n3[2])
if ds:
    print(f"{n3[0]} is d-separated from {n3[1]} | {n3[2]}")
else:
    print(f"{n3[0]} is d-connected to {n3[1]} | {n3[2]}")

# Perform 10 random d-separation tests
for r in range(1, 11):
    n3 = random.sample(nodes, 3)
    ds = dsep(bn, n3[0], n3[1], n3[2])
    if ds:
        print(f"{n3[0]} is d-separated from {n3[1]} | {n3[2]}")
    else:
        print(f"{n3[0]} is d-connected to {n3[1]} | {n3[2]}")
        
if __name__ == "__main__":
    pass
