from models.baseline import build_model, cros_val_fit


def test_model(xy):
    X, y = xy

    model = build_model()
    model.fit(X.to_numpy(), y.to_numpy())
    cros_val_fit(model, X.to_numpy(), y.to_numpy(), X.to_numpy())
