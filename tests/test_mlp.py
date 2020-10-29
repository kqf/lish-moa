import pytest
from models.mlp import build_model, build_preprocessor


def test_preprocessor(xy):
    X, y = xy
    xtransformed = build_preprocessor().fit_transform(X.to_numpy(), y.to_numpy())
    import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa


@pytest.mark.skip
def test_model(xy):
    X, y = xy

    model = build_model()
    model.fit(X.to_numpy(), y.to_numpy())
