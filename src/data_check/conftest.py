import pytest
import pandas as pd
# import wandb # REMOVED

# DO NOT MODIFY
@pytest.fixture(scope="session")
def data(csv):
    # run = wandb.init(job_type="data_check_fixture") # REMOVED
    # artifact = run.use_artifact(csv) # REMOVED
    # local_path = artifact.file() # REMOVED
    local_path = csv # Simplified for no W&B interaction
    df = pd.read_csv(local_path)
    # run.finish() # REMOVED
    return df


# DO NOT MODIFY
@pytest.fixture(scope="session")
def ref_data(ref):
    # run = wandb.init(job_type="data_check_fixture") # REMOVED
    # artifact = run.use_artifact(ref) # REMOVED
    # local_path = artifact.file() # REMOVED
    local_path = ref # Simplified for no W&B interaction
    df = pd.read_csv(local_path)
    # run.finish() # REMOVED
    return df


# Fixture to provide kl_threshold to tests
@pytest.fixture(scope="session")
def kl_threshold(kl_threshold_arg):
    return kl_threshold_arg

# Fixture to provide min_price to tests
@pytest.fixture(scope="session")
def min_price(min_price_arg):
    return min_price_arg

# Fixture to provide max_price to tests
@pytest.fixture(scope="session")
def max_price(max_price_arg):
    return max_price_arg


# Define command line arguments for pytest
def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default="clean_sample.csv:latest", help="Path to the current CSV data")
    parser.addoption("--ref", action="store", default="clean_sample.csv:reference", help="Path to the reference CSV data")
    parser.addoption("--kl_threshold", action="store", type=float, default=0.2, help="Kullback-Leibler divergence threshold")
    parser.addoption("--min_price", action="store", type=float, default=10.0, help="Minimum price for filtering outliers")
    parser.addoption("--max_price", action="store", type=float, default=350.0, help="Maximum price for filtering outliers")


# Provide fixtures from command line arguments
@pytest.fixture(scope="session")
def csv(request):
    return request.config.getoption("--csv")

@pytest.fixture(scope="session")
def ref(request):
    return request.config.getoption("--ref")

@pytest.fixture(scope="session")
def kl_threshold_arg(request):
    return request.config.getoption("--kl_threshold")

@pytest.fixture(scope="session")
def min_price_arg(request):
    return request.config.getoption("--min_price")

@pytest.fixture(scope="session")
def max_price_arg(request):
    return request.config.getoption("--max_price")