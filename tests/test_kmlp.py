import pytest
import numpy as np
from models.kmlp import build_model, build_preprocessor, cv_fit


@pytest.mark.skip
def test_preprocessor(xy):
    X, y = xy
    xtransformed = build_preprocessor().fit_transform(X, y)
    n_samples, n_featuers = xtransformed.shape

    assert n_samples == X.shape[0]
    assert n_featuers == 875


def test_model(xy):
    X = xy[0].to_numpy()
    y = xy[1].to_numpy().astype(np.float32)

    model = build_model()
    model.fit(X, y)


def test_cv_model(xy):
    X = xy[0].to_numpy()
    y = xy[1].to_numpy().astype(np.float32)

    cv_fit(build_model(), X, y, X)
