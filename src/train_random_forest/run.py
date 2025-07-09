#!/usr/bin/env python
"""
Train a Random Forest Regressor on the training data and save the model
"""
import argparse
import json
import logging
import os

import pandas as pd
# import wandb # REMOVED: W&B logging disabled due to persistent errors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow # Keep mlflow for model saving


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_inference_pipeline(rf_config):
    # Define categorical and numerical features based on the dataset
    non_ordinal_features = ["room_type", "neighbourhood_group"]
    numerical_features = ["minimum_nights", "number_of_reviews", "reviews_per_month",
                          "calculated_host_listings_count", "availability_365",
                          "latitude", "longitude"]
    
    non_ordinal_categorical_preproc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numerical_features), # numerical
            ("cat", non_ordinal_categorical_preproc, non_ordinal_features),  # non-ordinal categorical
        ],
        remainder="passthrough",
    )

    sk_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=rf_config["n_estimators"],
            max_depth=rf_config["max_depth"],
            random_state=rf_config["random_state"],
            **rf_config.get("regressor", {})
        ))
    ])

    return sk_pipe


def go(args):

    # W&B initialization and logging removed
    # run = wandb.init(project="nyc_airbnb", group="train_random_forest", job_type="train_random_forest")
    # run.config.update(args)

    logger.info("Fetching trainval_artifact: %s", args.trainval_artifact)
    # For now, assume trainval_artifact is a direct file path
    artifact_path = args.trainval_artifact # Simplified for no W&B interaction
    df = pd.read_csv(artifact_path)

    # Split the data into training and validation sets
    logger.info("Splitting data into training and validation sets (val_size=%s, random_seed=%s, stratify_by=%s)",
                args.val_size, args.random_seed, args.stratify_by)
    X_train, X_val, y_train, y_val = train_test_split(
        df.drop("price", axis=1),
        df["price"],
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Build the inference pipeline
    rf_config = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_seed,
        "min_samples_leaf": args.min_samples_leaf,
        "min_samples_split": args.min_samples_split,
        "regressor_name": args.regressor_name,
    }
    sk_pipe = get_inference_pipeline(rf_config)

    sk_pipe.fit(X_train, y_train)

    # Evaluate model on validation set
    predictions = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    logger.info(f"Val MAE: {mae:.2f}")
    logger.info(f"Val R2: {r2:.2f}")

    # Save the pipeline using MLFlow model export.
    mlflow.sklearn.save_model(sk_pipe, args.output_artifact)

    # W&B logging removed
    # run.log({"val_mae": mae, "val_r2": r2})
    # run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Name of the input trainval artifact",
        required=True
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Validation set size (e.g., 0.2 for 20%)",
        required=True
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for reproducibility",
        required=True
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to stratify by (e.g., 'neighbourhood_group' or 'none')",
        required=True
    )

    parser.add_argument(
        "--regressor_name",
        type=str,
        help="Name of the regressor (e.g., 'RandomForestRegressor')",
        required=True
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Number of trees in the random forest",
        required=True
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        help="Maximum depth of the trees",
        required=True
    )

    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        help="Minimum number of samples required to be at a leaf node",
        required=True
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        help="Minimum number of samples required to split an internal node",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output model artifact",
        required=True
    )

    parser.add_argument(
        "--rf_config",
        type=str,
        help="Path to the random forest configuration JSON file",
        required=True
    )

    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        help="Maximum number of TF-IDF features to use",
        required=True
    )

    args = parser.parse_args()

    go(args)
