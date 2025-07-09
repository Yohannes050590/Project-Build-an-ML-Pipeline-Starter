import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
import pandas as pd # Added this import for the data_split workaround

# NOTE: wandb import is removed as we are disabling W&B logging due to persistent errors
# import wandb

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    # Temporarily commenting out W&B environment variables to bypass connection/file errors
    # os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    # os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Get the root path of the project for local component paths
    root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory for MLflow runs
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Change the current working directory to the temporary directory
        # This is important for MLflow to correctly find artifacts and log paths
        old_cwd = os.getcwd()
        os.chdir(tmp_dir)

        if "download" in active_steps:
            # Download file and load in W&B
            # W&B logging for this step is disabled via 'wandb_project: none' in config
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                    # "wandb_project": "none", # Removed, as get_data component doesn't support this param
                },
            )

        if "basic_cleaning" in active_steps:
            # Basic cleaning step: cleans the raw data and creates a new clean_sample.csv artifact
            _ = mlflow.run(
                os.path.join(root_path, "src", "basic_cleaning"), # Path to local basic_cleaning component
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Data with common cleaning applied",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            # Data check step: performs tests on the cleaned data
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"), # Path to local data_check component
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference", # Assumes 'reference' tag has been manually added in W&B
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"], # Use etl's min_price for consistency
                    "max_price": config["etl"]["max_price"], # Use etl's max_price for consistency
                },
            )

        if "data_split" in active_steps:
            # Data split step: segregates data into training, validation, and test sets
            # Temporarily bypassing remote component due to GLIBCXX compatibility issues
            # Manually creating placeholder files as W&B logging is problematic
            
            # Create dummy trainval_data.csv
            trainval_df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})
            trainval_output_path = "trainval_data.csv"
            trainval_df.to_csv(trainval_output_path, index=False)

            # Create dummy test_data.csv
            test_df = pd.DataFrame({'col1': [7,8], 'col2': [9,10]})
            test_output_path = "test_data.csv"
            test_df.to_csv(test_output_path, index=False)


        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            # This is passed as a file to the train_random_forest component
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp) # DO NOT TOUCH

            # Train random forest step: trains the model and logs it as an artifact
            _ = mlflow.run(
                os.path.join(root_path, "src", "train_random_forest"), # Path to local train_random_forest component
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "regressor_name": config["modeling"]["random_forest"]["regressor_name"],
                    "n_estimators": config["modeling"]["random_forest"]["n_estimators"],
                    "max_depth": config["modeling"]["random_forest"]["max_depth"],
                    "min_samples_leaf": config["modeling"]["random_forest"]["min_samples_leaf"],
                    "min_samples_split": config["modeling"]["random_forest"]["min_samples_split"],
                    "output_artifact": "random_forest_export",
                    "rf_config": rf_config_path,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                },
            )

        if "test_regression_model" in active_steps:
            # Test regression model step: tests the best performing model against the test set
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model", # Path to remote component
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod", # Assumes 'prod' alias has been manually added in W&B
                    "test_artifact": "test_data.csv:latest",
                },
            )
        
        # Change back to the original working directory
        os.chdir(old_cwd)


if __name__ == "__main__":
    go()
