import pytest
from models.mlp import build_model, build_preprocessor, cv_fit


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
