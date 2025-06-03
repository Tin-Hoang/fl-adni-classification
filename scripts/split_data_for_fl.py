#!/usr/bin/env python3
"""Script to split data into train and validation sets for Federated Learning.

This script takes a CSV file containing data from multiple sites and splits it into train and validation datasets
for use in a Federated Learning setup. It distributes the sites among a specified number of clients,
then splits each client's data into train and validation sets based on a given ratio.
The main functionality is encapsulated in the `FederatedDataSplitter` class.

Command-line Arguments:
    csv_file (str): Path to the input CSV file (required positional argument).
    --output-dir (str): Directory to store output CSV files.
                        Defaults to the same directory as the input CSV if not specified.
    --num-clients (int): Number of federated learning clients.
                         Defaults to 2.
    --site-col (str): Column name for site identification.
                      Defaults to "SITE".
    --train-ratio (float): Ratio for the train/validation split (e.g., 0.8 for 80% train).
                           Must be between 0 and 1. Defaults to 0.8.
    --seed (int): Random seed for reproducibility of shuffling and splitting.
                  Defaults to 42.
    --log-level (str): Set the logging level.
                       Choices: DEBUG, INFO, WARNING, ERROR. Defaults to INFO.
"""

import argparse
import logging
import os
import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
from collections import Counter

# Configure basic logging (will be enhanced in main)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(log_level, output_dir):
    """Set up logging to console and file.

    Args:
        log_level: Logging level
        output_dir: Directory to store log file
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"split_data_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_file}")


class FederatedDataSplitter:
    """Splits data for Federated Learning based on sites."""

    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        num_clients: int,
        site_col: str = "SITE",
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """Initialize with input parameters.

        Args:
            csv_path: Path to the input CSV file
            output_dir: Directory to store output CSV files
            num_clients: Number of federated learning clients
            site_col: Column name for site identification
            train_ratio: Ratio for train/validation split
            seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.num_clients = num_clients
        self.site_col = site_col
        self.train_ratio = train_ratio
        self.seed = seed
        self.df = None

        # Extract the base name from the input CSV file (without extension)
        self.csv_basename = Path(csv_path).stem
        logger.info(f"Using CSV basename '{self.csv_basename}' as prefix for output files")

        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def load_data(self) -> None:
        """Load data from CSV file."""
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, low_memory=False)

        if self.site_col not in self.df.columns:
            raise ValueError(f"Site column '{self.site_col}' not found in CSV")

        # Convert site column to integer type
        self.df[self.site_col] = self.df[self.site_col].astype(int)

        logger.info(f"Loaded {len(self.df)} records from {len(self.df[self.site_col].unique())} sites")

    def distribute_sites(self) -> Dict[int, List[int]]:
        """Distribute sites among clients to balance the number of records.

        Returns:
            Dictionary mapping client IDs to lists of site IDs
        """
        # Count records per site
        site_counts = Counter(self.df[self.site_col])
        logger.info(f"Site distribution: {dict(site_counts)}")

        # Sort sites by number of records (descending)
        sorted_sites = sorted(site_counts.items(), key=lambda x: x[1], reverse=True)

        # Initialize client-to-sites mapping
        client_sites = {i+1: [] for i in range(self.num_clients)}
        client_record_counts = {i+1: 0 for i in range(self.num_clients)}

        # Distribute sites using a greedy approach - assign each site to the client with fewest records
        for site, count in sorted_sites:
            # Find client with fewest records
            min_client = min(client_record_counts, key=client_record_counts.get)
            client_sites[min_client].append(site)
            client_record_counts[min_client] += count

        # Log the distribution
        for client, sites in client_sites.items():
            site_counts = [self.df[self.df[self.site_col] == site].shape[0] for site in sites]
            logger.info(f"Client {client}: {len(sites)} sites, {sum(site_counts)} records")
            logger.info(f"  Sites: {', '.join(map(str, sites))}")

        return client_sites

    def split_and_save(self, client_sites: Dict[int, List[int]]) -> None:
        """Split data for each client into train and validation sets and save to CSV.

        Args:
            client_sites: Dictionary mapping client IDs to lists of site IDs
        """
        np.random.seed(self.seed)

        # Initialize DataFrames to collect all train and val data
        all_train_df = pd.DataFrame()
        all_val_df = pd.DataFrame()

        for client_id, sites in client_sites.items():
            # Get data for current client
            client_df = self.df[self.df[self.site_col].isin(sites)]
            logger.info(f"Client {client_id} has {len(client_df)} records")

            # Random shuffle
            client_df = client_df.sample(frac=1, random_state=self.seed)

            # Split into train and validation
            split_idx = int(len(client_df) * self.train_ratio)
            train_df = client_df.iloc[:split_idx]
            val_df = client_df.iloc[split_idx:]

            # Collect data for combined datasets
            all_train_df = pd.concat([all_train_df, train_df])
            all_val_df = pd.concat([all_val_df, val_df])

            logger.info(f"Client {client_id}: {len(train_df)} train, {len(val_df)} validation records")

            # Save to CSV with original filename prefix and record count
            train_path = self.output_dir / f"{self.csv_basename}_client_{client_id}_train_{len(train_df)}images.csv"
            val_path = self.output_dir / f"{self.csv_basename}_client_{client_id}_val_{len(val_df)}images.csv"

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)

            logger.info(f"Saved client {client_id} data to {train_path} and {val_path}")

        # Save combined datasets with original filename prefix
        all_train_path = self.output_dir / f"{self.csv_basename}_client_all_train_{len(all_train_df)}images.csv"
        all_val_path = self.output_dir / f"{self.csv_basename}_client_all_val_{len(all_val_df)}images.csv"

        all_train_df.to_csv(all_train_path, index=False)
        all_val_df.to_csv(all_val_path, index=False)

        logger.info(f"Saved combined data: {len(all_train_df)} train, {len(all_val_df)} validation records")
        logger.info(f"Combined files: {all_train_path} and {all_val_path}")

    def run(self) -> None:
        """Run the data splitting process."""
        self.load_data()
        client_sites = self.distribute_sites()
        self.split_and_save(client_sites)
        logger.info("Data splitting completed successfully")


def main():
    """Main function to run the data splitter."""
    parser = argparse.ArgumentParser(description="Split data for Federated Learning based on sites")

    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to store output CSV files (default: same directory as input CSV)"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of federated learning clients (default: 2)"
    )

    parser.add_argument(
        "--site-col",
        type=str,
        default="SITE",
        help="Column name for site identification (default: SITE)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio for train/validation split (default: 0.8)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Validate input paths exist
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return 1

    # Set output directory to input directory if not specified
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.csv_file))
        logger.info(f"Output directory not specified, using input directory: {output_dir}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging to console and file
    setup_logging(args.log_level, output_dir)

    # Log arguments
    logger.info("Script arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Validate num_clients is at least 1
    if args.num_clients < 1:
        logger.error("Number of clients must be at least 1")
        return 1

    # Validate train_ratio is between 0 and 1
    if not 0 < args.train_ratio < 1:
        logger.error("Train ratio must be between 0 and 1")
        return 1

    try:
        # Initialize and run the data splitter
        splitter = FederatedDataSplitter(
            csv_path=args.csv_file,
            output_dir=output_dir,
            num_clients=args.num_clients,
            site_col=args.site_col,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        splitter.run()
        return 0
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
