import pytest
import numpy as np
from models.baseline import read_data


@pytest.fixture
def xy():
    train = read_data("data/train_features.csv")
    targets = read_data("data/train_targets_scored.csv")
    for i, _ in enumerate(targets.columns):
        targets.iloc[:, i] = np.random.binomial(1, 0.5, targets.shape[0])
    return train.head(100), targets.head(100)
