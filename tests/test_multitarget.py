import pytest
from models.multitarget import build_model, cros_val_fit


@pytest.mark.skip("It should not work")
def test_model(xy):
    X, y = xy

    model = build_model()
    model.fit(X, y)
    cros_val_fit(model, X, y, X)
