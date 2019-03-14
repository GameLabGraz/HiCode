import os
import networkx as nx
import community
import copy
from itertools import combinations
from typing import Dict, List
from util.timing import time_f as timeit
import util.pickle_tools

Partition = Dict[int, int]
Community = List[int]


def hex_id(obj: object) -> str:
    return hex(id(obj))


def collapse_partition(partition: Partition) -> List[Community]:
    """
    Reduces the node -> community dictionary to a list of communities.
    A community is a list of nodes.
    :param partition: The partition (Dict[node, community]) to collapse.
    :return: A list of lists with each element representing a community.
    """
    community_list = [[k for k, v in partition.items() if v == value] for value in range(max(partition.values()) + 1)]

    for l1, l2 in combinations(community_list, 2):
        assert len(set(l1).intersection(l2)) == 0, "Communities are not disjoint"
    return community_list


def weighted_degree_sum(graph: nx.Graph, nbunch=None) -> float:
    """
    Calculates the sum of degrees for some or all nodes in a graph.
    :param graph: The graph containing nodes and edges.
    :param nbunch: (optional) Nodes to consider during calculation.
    :return: The weighted sum of degrees.
    """
    weight_sum = 0.0
    for node, degree in graph.degree(nbunch, weight='weight'):
        weight_sum += degree
    return weight_sum


class HiCode:
    """HiCode instance"""
    def __init__(self, graph: nx.Graph, num_layers: int, output_dir: str = "out_hicode"):
        self.num_layers = num_layers
        self.graph = copy.deepcopy(graph)
        self.layers: List[HiCodeLayer] = []
        self.output_dir = os.path.join(output_dir, str(num_layers))

    def identify(self):
        graph_reduced = self.graph.copy()

        for layer_idx in range(self.num_layers):
            print(f"{self.num_layers} - layer: {layer_idx}")
            partition = community.best_partition(graph_reduced)
            print(f"{self.num_layers} - Found {len(collapse_partition(partition))} communities in layer: {layer_idx}.")

            # todo: determine whether to store graph or reduce graph
            hicode_layer = HiCodeLayer(layer_idx, graph_reduced, partition)
            hicode_layer.reduce_weights_with_partition(partition)

            hicode_layer.calculate_modularity()
            self.layers.append(hicode_layer)

            graph_reduced = hicode_layer.graph

    def refine(self, num_iterations: int):
        for refinement_iteration in range(1, num_iterations + 1):
            print(f"{self.num_layers} - Refinement Iteration: {refinement_iteration}")
            for i, layer_i in enumerate(self.layers):
                for k, layer_k in enumerate(self.layers):
                    if i == k:
                        print(f"{self.num_layers} Iteration: {refinement_iteration}: {i} == {k}")
                        continue

                    print(f"\t{self.num_layers} Partition {k} ({hex_id(layer_k.graph)})"
                          f" -> {i} ({hex_id(layer_i.graph)})")
                    layer_i.refine_with(layer_k)
                layer_i.calculate_modularity()

    def write_layers_to_file(self):
        os.makedirs(self.output_dir)

        for i, layer in enumerate(self.layers):
            util.pickle_tools.save_obj(layer, f"layer{i}", self.output_dir)

    def run_layer_detection(self, refinement_iterations: int):
        self.identify()

        # restore original graph
        for layer in self.layers:
            layer.graph = copy.deepcopy(self.graph)
            # print(hex_id(layer.graph))

        self.refine(refinement_iterations)
        self.write_layers_to_file()
        print(f"{self.num_layers} - FINISHED.")


class HiCodeLayer:
    """Class representing on layer in the HiCode algorithm"""

    def __init__(self, index: int, graph: nx.Graph, partition: Partition):
        self.graph = copy.deepcopy(graph)
        self.index = index
        self.partition = partition
        self.modularities: List[float] = []

    def get_q_tick(self, comm: Community) -> float:
        """
        Determines the factor q' used in "ReduceEdge" and "ReduceWeight" for a given community in graph.
        :param comm: Community (list-of-nodes) for which to calculate q'.
        :return: The reduction factor q'.
        """
        community_graph = self.graph.subgraph(comm)
        n = len(self.graph)  # number of nodes
        n_k = len(community_graph)

        assert 1 < n_k < n, "1 < n_k ((0}) <  N ({1})".format(n_k, n)

        # as each edge is counted twice - (i, j) & (j, i) - by {in, out}_degree
        e_kk_2 = weighted_degree_sum(community_graph)
        e_kk = e_kk_2 / 2.
        d_k = weighted_degree_sum(self.graph, comm)
        # assert 0 < e_kk_2 <= d_k, "0 < e_kk_2 ({0}) < d_k ({1}) 2 * e_kk".format(e_kk_2, d_k)

        p_k = e_kk / (0.5 * n_k * (n_k - 1))
        q_k = (d_k - e_kk_2) / (n_k * (n - n_k))
        if p_k <= 0:
            return 0

        q_k_tick = q_k / p_k

        # assert 0 < q_k_tick < 1, "q_k' ({0}) out of bounds with p_q = {1}, q_k = {2}".format(q_k_tick, p_k, q_k)
        return q_k_tick

    def reduce_weights_with_partition(self, partition: Partition) -> None:
        """
        Reduces weights according to a given partition.
        :param partition: The list of node lists.
        :return: None.
        """
        for i, comm in enumerate(collapse_partition(partition)):
            assert len(comm) < len(self.graph), "Community {0} covers entire graph.".format(i)
            self.reduce_weights_with_community(comm, self.get_q_tick(comm))

    def reduce_weights_with_community(self, comm: community, reduction_factor: float):
        """
        Reduces the edges of a given in the stored graph according to a given reduction.
        :param comm: The community to reduce.
        :param reduction_factor: The factor by which to reduce edges.
        :return: None
        """
        # todo: determine whether to remove edges...
        edges_to_remove = []

        for (u, v) in self.graph.subgraph(comm).edges():
            if reduction_factor > 0:
                self.graph[u][v]['weight'] *= reduction_factor
            else:
                edges_to_remove.append((u, v))

        self.graph.remove_edges_from(edges_to_remove)

    @timeit
    def refine_with(self, other: 'HiCodeLayer') -> None:
        self.reduce_weights_with_partition(other.partition)
        print(hex_id(self.graph))

    def calculate_modularity(self):
        self.modularities.append(community.modularity(self.partition, self.graph))
