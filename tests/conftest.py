import pytest
from models.baseline import read_data


@pytest.fixture
def xy(size=100):
    train = read_data("data/train_features.csv")[:size]
    targets = read_data("data/train_targets_scored.csv")[:size]
    return train, targets
