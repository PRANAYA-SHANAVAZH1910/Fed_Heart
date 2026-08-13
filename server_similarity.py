"""
Similarity-Based Clustered Federated Learning Server
=====================================================

Heart-Disease Federated Learning Project

This server extends the standard Flower FL setup with
similarity-based client clustering.

Pipeline:

    Global/Cluster Model
            |
            v
    Local Hospital Training
            |
            v
    Local Model Parameters
            |
            v
    Delta W = Local - Model Received
            |
            v
    Cosine Distance Matrix
            |
            v
    Agglomerative Clustering
            |
            v
    Cluster Assignment
            |
            v
    Weighted FedAvg inside each Cluster
            |
            v
    Cluster-Specific Models

Important:
    - Raw patient data never leaves the client.
    - Clustering is performed entirely on the server.
    - Clients send model parameters and sample counts.
    - The original server.py can remain as the FedAvg baseline.

Example:

    python server_similarity.py --rounds 10 --min-clients 6 --clusters 3

The client can continue to be run as:

    python client.py 1
    python client.py 2
    ...

"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Optional

import numpy as np
import flwr as fl

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Your existing PyTorch model
from client import HeartDiseaseModel


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("similarity_fl_server")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def weighted_average(
    metrics: list[tuple[int, Metrics]]
) -> Metrics:
    """
    Compute sample-weighted averages of client metrics.

    Example:

        Hospital 1:
            100 samples
            accuracy = 0.80

        Hospital 2:
            200 samples
            accuracy = 0.90

        weighted accuracy =
            (100*0.80 + 200*0.90) / 300

    """

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    keys = set()

    for _, client_metrics in metrics:
        keys.update(client_metrics.keys())

    aggregated: dict[str, float] = {}

    for key in keys:

        weighted_sum = 0.0

        for num_examples, client_metrics in metrics:

            if key not in client_metrics:
                continue

            weighted_sum += (
                num_examples
                * float(client_metrics[key])
            )

        aggregated[key] = (
            weighted_sum / total_examples
        )

    return aggregated


# ============================================================================
# MODEL PARAMETER UTILITIES
# ============================================================================

def copy_ndarrays(
    parameters: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Create a deep copy of model parameters.
    """

    return [
        np.copy(parameter)
        for parameter in parameters
    ]


def flatten_parameters(
    parameters: list[np.ndarray],
) -> np.ndarray:
    """
    Flatten all neural-network layers into one vector.

    Example:

        [
            layer_1,
            layer_2,
            layer_3
        ]

    becomes:

        [
            w1, w2, w3, ...
        ]

    This allows us to compare entire model updates.
    """

    if not parameters:
        return np.array([], dtype=np.float64)

    return np.concatenate(
        [
            np.asarray(layer, dtype=np.float64).ravel()
            for layer in parameters
        ]
    )


def compute_delta(
    local_parameters: list[np.ndarray],
    reference_parameters: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Compute:

        Delta W = W_local - W_reference

    The reference model is the model that the client
    actually received before local training.
    """

    if len(local_parameters) != len(reference_parameters):
        raise ValueError(
            "Local and reference models have "
            "different numbers of layers."
        )

    deltas = []

    for local, reference in zip(
        local_parameters,
        reference_parameters,
    ):

        if local.shape != reference.shape:
            raise ValueError(
                "Local and reference layer shapes differ: "
                f"{local.shape} vs {reference.shape}"
            )

        deltas.append(
            np.asarray(local, dtype=np.float64)
            - np.asarray(reference, dtype=np.float64)
        )

    return deltas


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def build_initial_parameters(
    input_dim: int,
) -> Parameters:
    """
    Build the initial global model parameters.

    This uses the same HeartDiseaseModel as client.py.
    """

    model = HeartDiseaseModel(
        input_dim=input_dim
    )

    ndarrays = [
        parameter.detach()
        .cpu()
        .numpy()
        .copy()
        for parameter in model.parameters()
    ]

    return ndarrays_to_parameters(
        ndarrays
    )


# ============================================================================
# SAMPLE-WEIGHTED FEDAVG
# ============================================================================

def fedavg_parameters(
    client_results: list[
        tuple[list[np.ndarray], int]
    ],
) -> list[np.ndarray]:
    """
    Perform standard sample-weighted FedAvg.

    Parameters
    ----------
    client_results:

        [
            (client_parameters, num_samples),
            ...
        ]

    Returns
    -------
    Aggregated model parameters.
    """

    if not client_results:
        raise ValueError(
            "Cannot perform FedAvg with no clients."
        )

    total_samples = sum(
        num_samples
        for _, num_samples in client_results
    )

    if total_samples <= 0:
        raise ValueError(
            "Total number of samples must be > 0."
        )

    num_layers = len(
        client_results[0][0]
    )

    aggregated = [
        np.zeros_like(
            client_results[0][0][layer],
            dtype=np.float64,
        )
        for layer in range(num_layers)
    ]

    for parameters, num_samples in client_results:

        weight = (
            num_samples / total_samples
        )

        for layer in range(num_layers):

            aggregated[layer] += (
                weight
                * parameters[layer]
            )

    return [
        parameter.astype(
            np.float32
        )
        for parameter in aggregated
    ]


# ============================================================================
# SIMILARITY CLUSTERING
# ============================================================================

def build_cosine_distance_matrix(
    vectors: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """
    Build a cosine-distance matrix.

    Parameters
    ----------
    vectors:

        {
            client_id: flattened_delta_w
        }

    Returns
    -------
    client_ids
    distance_matrix

    Distance interpretation:

        0.0
            identical direction

        larger value
            increasingly different directions

    """

    client_ids = list(vectors.keys())

    if len(client_ids) == 0:
        return [], np.empty((0, 0))

    matrix = np.vstack(
        [
            vectors[client_id]
            for client_id in client_ids
        ]
    )

    # pairwise_distances supports metric="cosine"
    distance_matrix = pairwise_distances(
        matrix,
        metric="cosine",
    )

    # Numerical safety
    distance_matrix = np.nan_to_num(
        distance_matrix,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # Force exact symmetry
    distance_matrix = (
        distance_matrix
        + distance_matrix.T
    ) / 2.0

    # Diagonal must be zero
    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    return client_ids, distance_matrix


def perform_agglomerative_clustering(
    client_vectors: dict[str, np.ndarray],
    num_clusters: int,
) -> tuple[
    dict[str, int],
    list[str],
    np.ndarray,
]:
    """
    Cluster clients based on cosine distance between
    their model updates.

    Method:

        Delta W
          |
          v
        Flatten
          |
          v
        Cosine Distance
          |
          v
        Agglomerative Clustering
          |
          v
        Cluster IDs

    Average linkage is used.

    """

    client_ids, distance_matrix = (
        build_cosine_distance_matrix(
            client_vectors
        )
    )

    num_clients = len(client_ids)

    if num_clients == 0:
        return {}, [], distance_matrix

    # Only one client -> one cluster
    if num_clients == 1:

        return (
            {
                client_ids[0]: 0
            },
            client_ids,
            distance_matrix,
        )

    # Prevent invalid cluster counts
    actual_clusters = min(
        max(1, num_clusters),
        num_clients,
    )

    # One cluster means ordinary FedAvg
    if actual_clusters == 1:

        return (
            {
                client_id: 0
                for client_id in client_ids
            },
            client_ids,
            distance_matrix,
        )

    # Current sklearn API:
    #
    # metric="precomputed"
    # linkage="average"
    #
    # This is exactly what we want because we already
    # computed the cosine distance matrix.

    clustering = AgglomerativeClustering(
        n_clusters=actual_clusters,
        metric="precomputed",
        linkage="average",
    )

    labels = clustering.fit_predict(
        distance_matrix
    )

    assignments = {
        client_id: int(label)
        for client_id, label
        in zip(client_ids, labels)
    }

    return (
        assignments,
        client_ids,
        distance_matrix,
    )


# ============================================================================
# SIMILARITY CLUSTERED FLOWER STRATEGY
# ============================================================================

class SimilarityClusteredStrategy(
    fl.server.strategy.Strategy
):
    """
    Similarity-based clustered federated learning strategy.

    Main responsibilities:

        1. Select clients.
        2. Send global/cluster-specific models.
        3. Receive local models.
        4. Compute Delta W.
        5. Perform similarity clustering.
        6. Perform FedAvg independently within clusters.
        7. Store cluster-specific models.
        8. Send cluster models in subsequent rounds.

    """

    def __init__(
        self,
        initial_parameters: Parameters,
        min_clients: int,
        num_clusters: int,
        recluster_interval: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        results_dir: str = "results_similarity",
    ) -> None:

        self.initial_parameters = (
            initial_parameters
        )

        self.min_clients = min_clients

        self.num_clusters = num_clusters

        self.recluster_interval = (
            recluster_interval
        )

        self.fraction_fit = fraction_fit

        self.fraction_evaluate = (
            fraction_evaluate
        )

        self.results_dir = results_dir

        os.makedirs(
            self.results_dir,
            exist_ok=True,
        )

        # ------------------------------------------------------------------
        # Current global model
        # ------------------------------------------------------------------

        self.global_parameters = (
            parameters_to_ndarrays(
                initial_parameters
            )
        )

        # ------------------------------------------------------------------
        # Cluster models
        #
        # cluster_id -> parameters
        # ------------------------------------------------------------------

        self.cluster_models: dict[
            int,
            list[np.ndarray]
        ] = {}

        # ------------------------------------------------------------------
        # Client -> cluster
        #
        # cid -> cluster_id
        # ------------------------------------------------------------------

        self.client_clusters: dict[
            str,
            int
        ] = {}

        # ------------------------------------------------------------------
        # Model actually sent to each client
        #
        # Needed because after clustering a client may train from
        # a cluster-specific model rather than the global model.
        # ------------------------------------------------------------------

        self.client_reference_models: dict[
            str,
            list[np.ndarray]
        ] = {}

        # ------------------------------------------------------------------
        # Most recent Delta W
        # ------------------------------------------------------------------

        self.client_deltas: dict[
            str,
            np.ndarray
        ] = {}

        # ------------------------------------------------------------------
        # Latest distance matrix
        # ------------------------------------------------------------------

        self.last_distance_matrix: Optional[
            np.ndarray
        ] = None

        self.last_distance_client_ids: list[
            str
        ] = []

        logger.info(
            "SimilarityClusteredStrategy initialized | "
            "min_clients=%d | clusters=%d | "
            "recluster_interval=%d",
            self.min_clients,
            self.num_clusters,
            self.recluster_interval,
        )

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:

        return self.initial_parameters

    # ======================================================================
    # CLIENT SAMPLING
    # ======================================================================

    def _sample_clients(
        self,
        client_manager: ClientManager,
    ) -> list[ClientProxy]:

        logger.info(
            "Waiting for %d clients to connect...",
            self.min_clients,
        )

        clients_ready = client_manager.wait_for(
            self.min_clients,
            timeout=3600,
        )

        if not clients_ready:
            logger.error(
                "Timed out while waiting for %d clients.",
                self.min_clients,
            )
            return []

        num_available = client_manager.num_available()

        logger.info(
            "%d clients are now connected.",
            num_available,
        )

        clients = client_manager.sample(
            num_clients=self.min_clients,
            min_num_clients=self.min_clients,
        )

        logger.info(
            "Selected %d clients for the round.",
            len(clients),
        )

        return clients

    # ======================================================================
    # CONFIGURE FIT
    # ======================================================================

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[
        tuple[ClientProxy, FitIns]
    ]:

        clients = self._sample_clients(
            client_manager
        )

        if not clients:
            return []

        fit_config = {
            "server_round": server_round
        }

        configured = []

        for client in clients:

            cid = str(client.cid)

            # --------------------------------------------------------------
            # Determine which model this client should receive.
            #
            # Before clustering:
            #
            #       global model
            #
            # After clustering:
            #
            #       client's cluster model
            # --------------------------------------------------------------

            if cid in self.client_clusters:

                cluster_id = (
                    self.client_clusters[cid]
                )

                if (
                    cluster_id
                    in self.cluster_models
                ):

                    model_for_client = (
                        self.cluster_models[
                            cluster_id
                        ]
                    )

                    logger.debug(
                        "Round %d | Client %s -> "
                        "Cluster %d model",
                        server_round,
                        cid,
                        cluster_id,
                    )

                else:

                    model_for_client = (
                        self.global_parameters
                    )

            else:

                model_for_client = (
                    self.global_parameters
                )

            model_copy = copy_ndarrays(
                model_for_client
            )

            self.client_reference_models[
                cid
            ] = copy_ndarrays(
                model_copy
            )

            fit_ins = FitIns(
                parameters=ndarrays_to_parameters(
                    model_copy
                ),
                config=fit_config,
            )

            configured.append(
                (client, fit_ins)
            )

        logger.info(
            "Round %d | Sending training instructions "
            "to %d clients",
            server_round,
            len(configured),
        )

        return configured

    # ======================================================================
    # COMPUTE DELTA W
    # ======================================================================

    def _compute_client_deltas(
        self,
        results: list[
            tuple[ClientProxy, FitRes]
        ],
    ) -> dict[str, np.ndarray]:

        deltas = {}

        for client, fit_res in results:

            cid = str(client.cid)

            local_parameters = (
                parameters_to_ndarrays(
                    fit_res.parameters
                )
            )

            reference_parameters = (
                self.client_reference_models.get(
                    cid
                )
            )

            if reference_parameters is None:

                logger.warning(
                    "No reference model stored for "
                    "client %s. Skipping Delta W.",
                    cid,
                )

                continue

            delta = compute_delta(
                local_parameters,
                reference_parameters,
            )

            flattened_delta = (
                flatten_parameters(
                    delta
                )
            )

            deltas[cid] = (
                flattened_delta
            )

            delta_norm = float(
                np.linalg.norm(
                    flattened_delta
                )
            )

            logger.info(
                "Client %s | Delta W dimension=%d | "
                "norm=%.6f",
                cid,
                len(flattened_delta),
                delta_norm,
            )

        return deltas

    # ======================================================================
    # CLUSTERING
    # ======================================================================

    def _should_recluster(
        self,
        server_round: int,
    ) -> bool:

        return (
            server_round > 0
            and
            server_round
            % self.recluster_interval
            == 0
        )

    def _cluster_clients(
        self,
        server_round: int,
        deltas: dict[str, np.ndarray],
    ) -> None:

        if not deltas:

            logger.warning(
                "No client Delta W vectors available "
                "for clustering."
            )

            return

        if len(deltas) < 2:

            logger.warning(
                "Need at least 2 clients for clustering. "
                "Using one cluster."
            )

            self.client_clusters = {
                cid: 0
                for cid in deltas
            }

            return

        (
            assignments,
            client_ids,
            distance_matrix,
        ) = perform_agglomerative_clustering(
            client_vectors=deltas,
            num_clusters=self.num_clusters,
        )

        self.client_clusters = assignments

        self.last_distance_matrix = (
            distance_matrix
        )

        self.last_distance_client_ids = (
            client_ids
        )

        # --------------------------------------------------------------
        # Log distance matrix
        # --------------------------------------------------------------

        self._log_distance_matrix(
            server_round
        )

        # --------------------------------------------------------------
        # Log clusters
        # --------------------------------------------------------------

        logger.info("")
        logger.info(
            "================================================"
        )
        logger.info(
            "ROUND %d — SIMILARITY CLUSTERING",
            server_round,
        )
        logger.info(
            "================================================"
        )

        clusters: dict[
            int,
            list[str]
        ] = {}

        for cid, cluster_id in assignments.items():

            clusters.setdefault(
                cluster_id,
                []
            ).append(cid)

        for cluster_id in sorted(
            clusters.keys()
        ):

            members = clusters[
                cluster_id
            ]

            logger.info(
                "Cluster %d: %s",
                cluster_id,
                ", ".join(
                    sorted(members)
                ),
            )

        logger.info(
            "================================================"
        )

        # Save assignments
        self._save_cluster_assignments(
            server_round
        )

    # ======================================================================
    # DISTANCE MATRIX LOGGING
    # ======================================================================

    def _log_distance_matrix(
        self,
        server_round: int,
    ) -> None:

        if (
            self.last_distance_matrix is None
            or not self.last_distance_client_ids
        ):

            return

        logger.info(
            "Cosine distance matrix:"
        )

        matrix = (
            self.last_distance_matrix
        )

        ids = (
            self.last_distance_client_ids
        )

        header = (
            "       "
            + " ".join(
                f"{cid:>10}"
                for cid in ids
            )
        )

        logger.info(header)

        for i, cid in enumerate(ids):

            row = (
                f"{cid:>6} "
                + " ".join(
                    f"{matrix[i, j]:10.4f}"
                    for j in range(len(ids))
                )
            )

            logger.info(row)

    # ======================================================================
    # SAVE CLUSTER ASSIGNMENTS
    # ======================================================================

    def _save_cluster_assignments(
        self,
        server_round: int,
    ) -> None:

        path = os.path.join(
            self.results_dir,
            "cluster_assignments.csv",
        )

        file_exists = os.path.exists(
            path
        )

        with open(
            path,
            "a",
            newline="",
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow(
                    [
                        "round",
                        "client_id",
                        "cluster_id",
                    ]
                )

            for cid, cluster_id in sorted(
                self.client_clusters.items()
            ):

                writer.writerow(
                    [
                        server_round,
                        cid,
                        cluster_id,
                    ]
                )

    # ======================================================================
    # SAVE DISTANCE MATRIX
    # ======================================================================

    def _save_distance_matrix(
        self,
        server_round: int,
    ) -> None:

        if self.last_distance_matrix is None:

            return

        filename = (
            f"distance_matrix_round_"
            f"{server_round}.csv"
        )

        path = os.path.join(
            self.results_dir,
            filename,
        )

        ids = (
            self.last_distance_client_ids
        )

        with open(
            path,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                ["client_id"] + ids
            )

            for i, cid in enumerate(ids):

                writer.writerow(
                    [
                        cid,
                        *[
                            float(
                                self.last_distance_matrix[
                                    i, j
                                ]
                            )
                            for j in range(
                                len(ids)
                            )
                        ],
                    ]
                )

    # ======================================================================
    # CLUSTER-WISE FEDAVG
    # ======================================================================

    def _aggregate_cluster_models(
        self,
        results: list[
            tuple[ClientProxy, FitRes]
        ],
    ) -> dict[
        int,
        list[np.ndarray]
    ]:

        cluster_results: dict[
            int,
            list[
                tuple[
                    list[np.ndarray],
                    int
                ]
            ]
        ] = {}

        for client, fit_res in results:

            cid = str(client.cid)

            cluster_id = (
                self.client_clusters.get(
                    cid,
                    0
                )
            )

            parameters = (
                parameters_to_ndarrays(
                    fit_res.parameters
                )
            )

            num_samples = int(
                fit_res.num_examples
            )

            cluster_results.setdefault(
                cluster_id,
                []
            ).append(
                (
                    parameters,
                    num_samples,
                )
            )

        cluster_models = {}

        for cluster_id, client_results in (
            cluster_results.items()
        ):

            cluster_models[
                cluster_id
            ] = fedavg_parameters(
                client_results
            )

            total_samples = sum(
                samples
                for _, samples
                in client_results
            )

            logger.info(
                "Cluster %d | clients=%d | "
                "samples=%d",
                cluster_id,
                len(client_results),
                total_samples,
            )

        return cluster_models

    # ======================================================================
    # GLOBAL FEDAVG
    # ======================================================================

    def _aggregate_global_model(
        self,
        results: list[
            tuple[ClientProxy, FitRes]
        ],
    ) -> list[np.ndarray]:

        client_results = []

        for _, fit_res in results:

            parameters = (
                parameters_to_ndarrays(
                    fit_res.parameters
                )
            )

            client_results.append(
                (
                    parameters,
                    int(
                        fit_res.num_examples
                    ),
                )
            )

        return fedavg_parameters(
            client_results
        )

    # ======================================================================
    # AGGREGATE FIT
    # ======================================================================

    def aggregate_fit(
        self,
        server_round: int,
        results: list[
            tuple[ClientProxy, FitRes]
        ],
        failures,
    ) -> tuple[
        Optional[Parameters],
        dict[str, Scalar],
    ]:

        if not results:

            logger.warning(
                "Round %d received no successful "
                "client results.",
                server_round,
            )

            return None, {}

        logger.info(
            "Round %d | received %d client updates",
            server_round,
            len(results),
        )

        # --------------------------------------------------------------
        # STEP 1
        #
        # Compute Delta W
        # --------------------------------------------------------------

        deltas = (
            self._compute_client_deltas(
                results
            )
        )

        self.client_deltas = deltas

        # --------------------------------------------------------------
        # STEP 2
        #
        # Determine whether this is a clustering round.
        # --------------------------------------------------------------

        if self._should_recluster(
            server_round
        ):

            self._cluster_clients(
                server_round,
                deltas,
            )

            self._save_distance_matrix(
                server_round
            )

        # --------------------------------------------------------------
        # STEP 3
        #
        # If clusters exist, perform cluster-wise FedAvg.
        #
        # Otherwise perform normal global FedAvg.
        # --------------------------------------------------------------

        if self.client_clusters:

            self.cluster_models = (
                self._aggregate_cluster_models(
                    results
                )
            )

            # ----------------------------------------------------------
            # Also compute a global model.
            #
            # This is useful for:
            #   - logging
            #   - baseline comparison
            #   - clients without cluster assignments
            # ----------------------------------------------------------

            self.global_parameters = (
                self._aggregate_global_model(
                    results
                )
            )

            logger.info(
                "Round %d | cluster-wise FedAvg complete.",
                server_round,
            )

            metrics = {
                "num_clusters": float(
                    len(
                        self.cluster_models
                    )
                ),
                "num_clients": float(
                    len(results)
                ),
            }

        else:

            # No clustering yet
            self.global_parameters = (
                self._aggregate_global_model(
                    results
                )
            )

            metrics = {
                "num_clusters": 1.0,
                "num_clients": float(
                    len(results)
                ),
            }

            logger.info(
                "Round %d | standard FedAvg complete.",
                server_round,
            )

        # --------------------------------------------------------------
        # Return global model to Flower.
        #
        # IMPORTANT:
        #
        # configure_fit() will use cluster_models for the next
        # round, so the returned global model is mainly Flower's
        # server-side state and fallback model.
        # --------------------------------------------------------------

        return (
            ndarrays_to_parameters(
                self.global_parameters
            ),
            metrics,
        )

    # ======================================================================
    # CONFIGURE EVALUATION
    # ======================================================================

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[
        tuple[ClientProxy, EvaluateIns]
    ]:

        if self.fraction_evaluate <= 0:

            return []

        num_available = (
            client_manager.num_available()
        )

        if num_available < self.min_clients:

            return []

        num_clients = max(
            self.min_clients,
            int(
                np.ceil(
                    self.fraction_evaluate
                    * num_available
                )
            ),
        )

        num_clients = min(
            num_clients,
            num_available,
        )

        clients = client_manager.sample(
            num_clients=num_clients,
            min_num_clients=self.min_clients,
        )

        configured = []

        for client in clients:

            cid = str(client.cid)

            if (
                cid in self.client_clusters
                and
                self.client_clusters[cid]
                in self.cluster_models
            ):

                cluster_id = (
                    self.client_clusters[cid]
                )

                evaluation_model = (
                    self.cluster_models[
                        cluster_id
                    ]
                )

            else:

                evaluation_model = (
                    self.global_parameters
                )

            evaluate_ins = EvaluateIns(
                parameters=ndarrays_to_parameters(
                    copy_ndarrays(
                        evaluation_model
                    )
                ),
                config={
                    "server_round":
                    server_round
                },
            )

            configured.append(
                (
                    client,
                    evaluate_ins,
                )
            )

        return configured

    # ======================================================================
    # AGGREGATE EVALUATION
    # ======================================================================

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[
            tuple[ClientProxy, EvaluateRes]
        ],
        failures,
    ) -> tuple[
        Optional[float],
        dict[str, Scalar],
    ]:

        if not results:

            return None, {}

        total_samples = sum(
            int(
                result.num_examples
            )
            for _, result in results
        )

        if total_samples == 0:

            return None, {}

        weighted_loss = sum(
            result.loss
            * int(result.num_examples)
            for _, result in results
        )

        average_loss = (
            weighted_loss
            / total_samples
        )

        # --------------------------------------------------------------
        # Aggregate client-reported metrics
        # --------------------------------------------------------------

        metric_list = []

        for _, result in results:

            metric_list.append(
                (
                    int(
                        result.num_examples
                    ),
                    result.metrics,
                )
            )

        metrics = weighted_average(
            metric_list
        )

        logger.info(
            "Round %d | evaluation loss=%.6f | "
            "metrics=%s",
            server_round,
            average_loss,
            metrics,
        )

        return (
            float(average_loss),
            metrics,
        )

    # ======================================================================
    # CENTRALIZED EVALUATION
    # ======================================================================

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[
        tuple[
            float,
            dict[str, Scalar]
        ]
    ]:

        # We don't have a centralized dataset on the server.
        #
        # Patient data remains at hospitals.
        #
        # Therefore centralized server evaluation is disabled.

        return None


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Similarity-based clustered "
            "federated learning server."
        )
    )

    parser.add_argument(
        "--address",
        type=str,
        default="10.2.80.71:8080",
        help=(
            "Server address in HOST:PORT format."
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help=(
            "Number of federated learning rounds."
        ),
    )

    parser.add_argument(
        "--min-clients",
        type=int,
        default=5,
        help=(
            "Minimum number of hospitals "
            "required for each round."
        ),
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=2,
        help=(
            "Number of similarity-based "
            "clusters."
        ),
    )

    parser.add_argument(
        "--recluster-interval",
        type=int,
        default=5,
        help=(
            "Recompute client clusters every "
            "N rounds."
        ),
    )

    parser.add_argument(
        "--input-dim",
        type=int,
        default=15,
        help=(
            "Number of model input features."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_similarity",
        help=(
            "Directory used to save clustering "
            "results."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main(
    server_address: str,
    num_rounds: int,
    min_clients: int,
    num_clusters: int,
    recluster_interval: int,
    input_dim: int,
    results_dir: str,
) -> None:

    # ------------------------------------------------------------------------
    # Validate configuration
    # ------------------------------------------------------------------------

    if num_rounds <= 0:

        raise ValueError(
            "--rounds must be greater than 0."
        )

    if min_clients <= 0:

        raise ValueError(
            "--min-clients must be greater than 0."
        )

    if num_clusters <= 0:

        raise ValueError(
            "--clusters must be greater than 0."
        )

    if recluster_interval <= 0:

        raise ValueError(
            "--recluster-interval must be "
            "greater than 0."
        )

    # ------------------------------------------------------------------------
    # Initial model
    # ------------------------------------------------------------------------

    initial_parameters = (
        build_initial_parameters(
            input_dim
        )
    )

    # ------------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------------

    strategy = (
        SimilarityClusteredStrategy(
            initial_parameters=initial_parameters,
            min_clients=min_clients,
            num_clusters=num_clusters,
            recluster_interval=recluster_interval,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            results_dir=results_dir,
        )
    )

    # ------------------------------------------------------------------------
    # Server startup
    # ------------------------------------------------------------------------

    logger.info("")
    logger.info(
        "================================================"
    )
    logger.info(
        "SIMILARITY-BASED CLUSTERED FEDERATED LEARNING"
    )
    logger.info(
        "================================================"
    )
    logger.info(
        "Address            : %s",
        server_address,
    )
    logger.info(
        "Rounds             : %d",
        num_rounds,
    )
    logger.info(
        "Minimum clients    : %d",
        min_clients,
    )
    logger.info(
        "Clusters           : %d",
        num_clusters,
    )
    logger.info(
        "Recluster interval : %d rounds",
        recluster_interval,
    )
    logger.info(
        "Input dimensions   : %d",
        input_dim,
    )
    logger.info(
        "Results directory  : %s",
        results_dir,
    )
    logger.info(
        "================================================"
    )

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(
            num_rounds=num_rounds
        ),
        strategy=strategy,
    )

    logger.info("")
    logger.info(
        "================================================"
    )
    logger.info(
        "TRAINING COMPLETE"
    )
    logger.info(
        "================================================"
    )

    logger.info(
        "Loss history: %s",
        history.losses_distributed,
    )

    logger.info(
        "Metrics history: %s",
        history.metrics_distributed,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    args = parse_args()

    main(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        num_clusters=args.clusters,
        recluster_interval=args.recluster_interval,
        input_dim=args.input_dim,
        results_dir=args.results_dir,
    )