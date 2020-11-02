import pytest
from models.kmlp import read_data


@pytest.fixture
def xy(size):
    train = read_data("data/train_features.csv")[:size]
    targets = read_data("data/train_targets_scored.csv")[:size]
    return train, targets
