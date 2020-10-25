import pytest
import numpy as np
from models.baseline import read_data


@pytest.fixture
def xy():
    train = read_data("data/train_features.csv").head(100)
    targets = read_data("data/train_targets_scored.csv").head(100)

    # Generate the data to ensure positive labels
    for i, _ in enumerate(targets.columns):
        targets.iloc[:, i] = np.random.binomial(2, 0.5, targets.shape[0])

    return train, targets
