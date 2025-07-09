#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd
# import wandb # REMOVED: W&B logging disabled due to persistent errors


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# DO NOT MODIFY
def go(args):

    # W&B initialization and logging removed
    # run = wandb.init(project="nyc_airbnb", group="basic_cleaning", job_type="basic_cleaning")
    # run.config.update(args)

    # Download input artifact. For now, assume input_artifact is a direct file path.
    # MLflow handles fetching artifacts into the temporary run directory.
    logger.info("Fetching input artifact: %s", args.input_artifact)
    # artifact = run.use_artifact(args.input_artifact)
    # artifact_path = artifact.file()
    artifact_path = args.input_artifact # Simplified for no W&B interaction
    df = pd.read_csv(artifact_path)

    # Drop outliers based on min_price and max_price arguments
    logger.info("Dropping price outliers: min_price=%s, max_price=%s", args.min_price, args.max_price)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime, handling errors by coercing to NaT
    logger.info("Converting 'last_review' to datetime format")
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # TODO: Fix the problem with the geolocation of some of the points that are outside of NYC
    # This will be added later in Phase 6, but we'll put the placeholder here now
    # idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    # df = df[idx].copy()

    # Save the cleaned data to a local CSV file
    output_path = f"{args.output_artifact}.csv" # Use output_artifact name for the file
    df.to_csv(output_path, index=False)
    logger.info("Cleaned data saved to %s", output_path)

    # W&B artifact logging removed
    # artifact = wandb.Artifact(
    #     name=args.output_artifact,
    #     type=args.output_type,
    #     description=args.output_description,
    # )
    # artifact.add_file(output_path)
    # run.log_artifact(artifact)
    # artifact.wait()

    # run.finish() # W&B run finish removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a dataset")

    # Arguments for input and output artifacts
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact to clean (e.g., 'sample.csv:latest')",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output cleaned artifact (e.g., 'clean_sample.csv')",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (e.g., 'clean_data')",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    # Arguments for data cleaning parameters
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for filtering outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for filtering outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)
