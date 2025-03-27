import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from itertools import count, islice
from typing import Dict, List, Set

from openai import OpenAI

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

#logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        # Helper Methods:

        def get_next_index():
            with counter_lock:
                return next(next_node_index_counter)

        def process_cluster_multithreaded(cluster, new_level_nodes, summarization_length, lock):
            node_texts = get_text(cluster)

            try:
                summarized_text = self.summarize(
                    context=node_texts,
                    max_tokens=summarization_length
                )
            except Exception as e:
                logging.error(f"Failed to summarize cluster: {e}")
                raise e


            # Safely get the next node index using the counter
            next_node_index = get_next_index()

            logging.info(
                f"Index {next_node_index}, Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Index {next_node_index}, Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        # Helper methods end.


        # construct_tree() start:
        logging.info("Using Cluster TreeBuilder")

        # Initialize a thread-safe counter starting at `new_node_index`
        next_node_index = len(all_tree_nodes)
        if use_multithreading:
            next_node_index_counter = count(start=next_node_index)  # Start the counter at `new_node_index`
            counter_lock = Lock()

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                try:
                    #with ThreadPoolExecutor(max_workers=5) as executor:
                    with ThreadPoolExecutor() as executor:
                        logging.info("Using multithreaded Summarization and Cluster processing")
                        futures = [
                            executor.submit(
                                process_cluster_multithreaded,
                                cluster,
                                new_level_nodes,
                                summarization_length,
                                lock
                            )
                            for cluster in clusters
                        ]

                        for future in as_completed(futures):
                            # check if a thread fails with exception
                            exception = future.exception()
                            if exception:
                                logging.error(f"Error in cluster processing: {exception}")

                                # Cancel remaining tasks
                                for f in futures:
                                    f.cancel()

                                # Propagate the exception to stop processing
                                raise exception

                except Exception as e:
                    logging.error(f"Aborting construct_tree due to error: {e}")
                    raise e # propagates error and aborts construct_tree

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
