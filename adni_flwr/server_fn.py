"""Utility functions for server-side federated learning."""

from typing import List, Tuple, Dict, Any, Optional, Union
from collections.abc import Mapping
import numpy as np

def safe_weighted_average(metrics: List[Tuple[int, Dict]], verbose: bool = True) -> Dict:
    """A safer implementation of weighted average that handles potential serialization issues.

    Processes scalar metrics (int, float) and passes through string metrics for JSON-encoded lists.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
        verbose: Whether to print debug information

    Returns:
        Aggregated metrics dictionary
    """
    if verbose:
        print(f"safe_weighted_average: Processing {len(metrics)} metric sets")

    # Early return for empty metrics
    if not metrics:
        return {}

    # Filter out empty or non-dict metrics
    filtered_metrics = []
    for i, (num_examples, metric_dict) in enumerate(metrics):
        if not isinstance(metric_dict, (dict, Mapping)) or not metric_dict:
            if verbose:
                print(f"Skipping metric set {i}: Not a valid dictionary")
            continue

        # Make a clean copy of the metrics dictionary with only serializable types
        clean_dict = {}
        for k, v in metric_dict.items():
            if not isinstance(k, str):
                if verbose:
                    print(f"Skipping non-string key {k} in metric set {i}")
                continue

            # Handle scalar values (numbers)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                clean_dict[k] = float(v)  # Convert all numbers to float for consistency
            # Handle string values (including JSON-encoded lists)
            elif isinstance(v, str) and k in ["predictions_json", "labels_json", "sample_info", "client_id"]:
                clean_dict[k] = v
                if verbose:
                    print(f"Keeping string metric '{k}' of length {len(v)}")
            elif verbose:
                print(f"Skipping non-scalar/non-string value for key {k} (type: {type(v).__name__})")

        if clean_dict:  # Only add if we have valid metrics
            filtered_metrics.append((num_examples, clean_dict))

    if verbose:
        print(f"After filtering: {len(filtered_metrics)} valid metric sets")

    if not filtered_metrics:
        return {}

    # Collect all unique metric names
    all_keys = set()
    for _, m in filtered_metrics:
        all_keys.update(m.keys())

    # Compute weighted average for each metric
    result = {}
    for key in all_keys:
        # Get all values for this key across clients
        values = []
        weights = []

        for num_examples, m in filtered_metrics:
            if key in m:
                value = m[key]

                # For scalar metrics, compute weighted average
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
                    weights.append(num_examples)
                # For string metrics (like predictions_json), just take the first one
                elif isinstance(value, str) and key in ["predictions_json", "labels_json", "sample_info", "client_id"]:
                    # Only add if we haven't seen this key yet
                    if key not in result:
                        result[key] = value
                        if verbose:
                            print(f"Using value from first client for string metric '{key}'")

        # Only compute average for scalar metrics if we have values
        if values and weights:
            # Convert to numpy arrays for vectorized computation
            values_array = np.array(values)
            weights_array = np.array(weights)

            # For training_time, use simple average instead of weighted average
            if key == "training_time":
                result[key] = float(np.mean(values_array))
                if verbose:
                    print(f"Computed simple average for training time: {result[key]:.4f} seconds")
            # If all weights are zero, use simple average
            elif np.sum(weights_array) == 0:
                result[key] = float(np.mean(values_array))
            else:
                # Compute weighted average
                result[key] = float(np.sum(values_array * weights_array) / np.sum(weights_array))

    if verbose:
        print(f"Aggregated {len(result)} metrics: {list(result.keys())}")

    return result
