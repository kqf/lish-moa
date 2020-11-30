import pytest
from models.pmlp import build_model
from models.pmlp import read_data


@pytest.fixture
def xy(size=1000):
    train = read_data("data/train_features.csv")[:size]
    targets = read_data("data/train_targets_scored.csv")[:size]
    return train, targets


def test_model(xy):
    X, y = xy
    model = build_model()
    model.fit(X, y)
