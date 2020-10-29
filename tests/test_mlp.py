from models.mlp import build_model


def test_model(xy):
    X, y = xy

    model = build_model()
    model.fit(X.to_numpy(), y.to_numpy())
