import pytest
from models.baseline import read_data


@pytest.fixture
def xy():
    train = read_data("data/train_features.csv")
    targets = read_data("data/train_targets_scored.csv")
    return train.head(100), targets.head(100)
