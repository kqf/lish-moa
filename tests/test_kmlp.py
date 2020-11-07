import pytest
from models.kmlp import build_model, build_preprocessor, cv_fit
from models.kmlp import read_data


@pytest.fixture
def xy(size=1000):
    train = read_data("data/train_features.csv")[:size]
    targets = read_data("data/train_targets_scored.csv")[:size]
    return train, targets


@pytest.mark.skip
def test_preprocessor(xy):
    X, y = xy
    xtransformed = build_preprocessor().fit_transform(X, y)
    n_samples, n_featuers = xtransformed.shape

    assert n_samples == X.shape[0]
    assert n_featuers == 875


def test_model(xy):
    X, y = xy
    model = build_model()
    model.fit(X, y)


def test_cv_model(xy):
    X, y = xy
    cv_fit(build_model(), X, y, X)
