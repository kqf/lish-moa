from models.baseline import build_model, cv_fit


def test_model(xy):
    X, y = xy
    model = build_model()
    model.fit(X, y)


def test_cv_model(xy):
    X, y = xy
    cv_fit(build_model(), X, y, X)
