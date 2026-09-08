"""
Flower server for the hospital heart-disease federated learning setup.

This server is designed to run against `client.py`, which defines
`HeartDiseaseModel` and `HospitalClient(fl.client.NumPyClient)`. Each
hospital process connects to this server at the configured address and
participates in FedAvg rounds.

Usage:
    python server.py
    python server.py --rounds 10 --min-clients 5

Then, in separate terminals (one per hospital):
    python client.py 1
    python client.py 2
    ...
"""


from __future__ import annotations

import argparse
import logging

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters

# NOTE: HeartDiseaseModel must be importable from client.py, and the
# __init__ typo (`_init_` -> `__init__`) must be fixed there first,
# otherwise this import (and the client itself) will fail.
from client  import HeartDiseaseModel
import copy
import numpy as np
from collections import defaultdict
logger = logging.getLogger("fl_server")
###############################################################################
# DELTA WEIGHT TRACKER
###############################################################################

class DeltaWeightTracker:
    """
    Computes

        ΔW = W_local - W_global

    for every client.

    This class DOES NOT perform clustering.

    It only observes client updates.

    Stage 1:
        • compute delta
        • flatten
        • store

    Later stages:

        • clustering
        • similarity
        • history
    """

    def __init__(self):

        self.global_weights = None

        self.client_delta = {}

    ###########################################################################

    def set_global_weights(self, global_weights):

        """
        Store current global model.
        """

        self.global_weights = [
            np.copy(layer)
            for layer in global_weights
        ]

    ###########################################################################

    def compute_delta(
        self,
        client_id,
        local_weights,
    ):

        """
        Compute

            ΔW

        for one client.
        """

        if self.global_weights is None:

            raise RuntimeError(
                "Global weights not initialized."
            )

        delta = []

        for local, global_layer in zip(
            local_weights,
            self.global_weights,
        ):

            delta.append(local - global_layer)

        self.client_delta[client_id] = delta

        return delta

    ###########################################################################

    @staticmethod
    def flatten(delta):

        """
        Convert

        list of layers

        →

        one vector.
        """

        return np.concatenate(
            [
                layer.flatten()
                for layer in delta
            ]
        )

    ###########################################################################

    def summary(self):

        """
        Print statistics.
        """

        logger.info("========== DELTA SUMMARY ==========")

        for client in self.client_delta:

            vector = self.flatten(
                self.client_delta[client]
            )

            logger.info(
                "Client %s | Dimension=%d | Norm=%0.4f",
                client,
                len(vector),
                np.linalg.norm(vector),
            )

        logger.info("===================================")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fl_server")


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate client-reported metrics using a sample-weighted average.

    Flower calls this after each fit/evaluate round with a list of
    (num_examples, metrics_dict) tuples -- one per participating client.
    This mirrors the num_samples-weighted averaging your custom
    FedAvgStrategy performed on raw parameters, but applied here to
    scalar metrics like "accuracy".

    Args:
        metrics: List of (num_examples, metrics_dict) pairs, one per
            client that reported metrics this round.

    Returns:
        A single metrics dict with the weighted-average value for each
        key present in the client metrics (e.g. {"accuracy": 0.83}).
    """
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    metric_keys = metrics[0][1].keys()

    for key in metric_keys:
        weighted_sum = sum(
            num_examples * client_metrics[key]
            for num_examples, client_metrics in metrics
            if key in client_metrics
        )
        aggregated[key] = weighted_sum / total_examples

    return aggregated


def build_initial_parameters(input_dim: int) -> fl.common.Parameters:
    """Build Flower Parameters from a freshly initialized global model.

    Providing explicit initial parameters (rather than letting Flower
    pull them from an arbitrary first client) ensures every client
    starts round 1 from the exact same weights.

    Args:
        input_dim: Number of input features the model expects. Must
            match the number of feature columns produced by
            load_hospital_data() in client.py (i.e. all columns except
            "TenYearCHD").

    Returns:
        A Flower Parameters object wrapping the initialized model's
        weights as NumPy arrays.
    """
    model = HeartDiseaseModel(input_dim=input_dim)
    ndarrays = [val.detach().cpu().numpy() for val in model.parameters()]
    return ndarrays_to_parameters(ndarrays)


def make_strategy(
    min_clients: int,
    input_dim: int,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
) -> fl.server.strategy.FedAvg:
    """Construct the FedAvg strategy used to coordinate training rounds.

    Args:
        min_clients: Minimum number of clients required to be
            connected, and required to participate in fit/evaluate,
            before a round proceeds. Set this to the number of
            hospitals you expect to have online (e.g. 5 or 10).
        input_dim: Number of input features, used to build the initial
            global model parameters.
        fraction_fit: Fraction of available clients sampled for
            training each round. 1.0 means "all connected clients
            train every round" (appropriate for a small, fixed set of
            hospitals rather than large-scale cross-device FL).
        fraction_evaluate: Fraction of available clients sampled for
            evaluation each round.

    Returns:
        A configured FedAvg strategy instance.
    """
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=build_initial_parameters(input_dim),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


def main(
    server_address: str,
    num_rounds: int,
    min_clients: int,
    input_dim: int,
) -> None:
    """Start the Flower server and run federated training.

    Args:
        server_address: Host:port the server listens on. Must match
            the address each HospitalClient connects to.
        num_rounds: Number of federated communication rounds to run.
        min_clients: Minimum number of hospital clients required
            before training starts.
        input_dim: Number of input features expected by
            HeartDiseaseModel (must match client.py's data).
    """
    strategy = make_strategy(min_clients=min_clients, input_dim=input_dim)
    tracker = DeltaWeightTracker()
    

    logger.info(
        "Starting Flower server at %s | rounds=%d | min_clients=%d | input_dim=%d",
        server_address,
        num_rounds,
        min_clients,
        input_dim,
    )

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    logger.info("Training complete.")
    logger.info("Loss history: %s", history.losses_distributed)
    logger.info("Metrics history: %s", history.metrics_distributed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the server entry point."""
    parser = argparse.ArgumentParser(description="Flower FL server for hospital data.")
    parser.add_argument(
        "--address",
        type=str,
        default="10.2.80.71:8080",
        help="Server address (host:port) to listen on.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds to run.",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=5,
        help="Minimum number of hospital clients required to start/run a round.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=15,
        help=(
            "Number of input features (columns excluding TenYearCHD). "
            "Must match client.py's load_hospital_data output."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        input_dim=args.input_dim,
    )