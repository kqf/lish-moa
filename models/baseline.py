# Inspired by:
# https://www.kaggle.com/fchmiel/xgboost-baseline-multilabel-classification

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold
from category_encoders import CountEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

from sklearn.multioutput import MultiOutputClassifier


SEED = 42
NFOLDS = 5
np.random.seed(SEED)


def build_model():
    # Former xgboost parameters
    params = {
        "colsample_bytree": 0.6522,
        "gamma": 3.6975,
        "learning_rate": 0.0503,
        "max_delta_step": 2.0706,
        "max_depth": 10,
        "min_child_weight": 31.5800,
        "n_estimators": 166,
        "subsample": 0.8639
    }

    model = make_pipeline(
        CountEncoder(cols=[0, 2]),
        MultiOutputClassifier(xgb.XGBClassifier(**params)),
    )

    return model


def read_data(path, ignore_col="sig_id"):
    df = pd.read_csv(path)
    df.drop(columns=[ignore_col], inplace=True)
    return df


def cros_val_fit(clf, X, y, X_test, cv=None):
    cv = cv or KFold(5)

    oof_preds = np.zeros(y.shape)
    test_preds = np.zeros((X_test.shape[0], y.shape[1]))
    oof_losses = []

    for fn, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print("Starting fold: ", fn)
        X_train, X_val = X[trn_idx], X[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # drop where cp_type==ctl_vehicle (baseline)
        ctl_mask = X_train[:, 0] == "ctl_vehicle"
        X_train = X_train[~ctl_mask, :]
        y_train = y_train[~ctl_mask]

        clf.fit(X_train, y_train)
        val_preds = clf.predict_proba(X_val)  # list of preds per class
        val_preds = np.array(val_preds)[:, :, 1].T  # take the positive class
        oof_preds[val_idx] = val_preds

        loss = log_loss(np.ravel(y_val), np.ravel(val_preds))
        oof_losses.append(loss)

        # preds = clf.predict_proba(X_test)
        # preds = np.array(preds)[:, :, 1].T  # take the positive class
        # test_preds += preds / NFOLDS

    return oof_losses,


def main():
    train = read_data("data/train_features.csv")
    targets = read_data("data/train_targets_scored.csv")

    test = read_data("data/test_features.csv")
    sub = read_data("data/sample_submission.csv")

    # drop id col
    X = train.to_numpy()
    X_test = test.to_numpy()
    y = targets.to_numpy()
    clf = build_model()

    cv = KFold(n_splits=NFOLDS)
    oof_losses, oof_preds, test_preds = cros_val_fit(clf, X, y, X_test, cv=cv)

    print(oof_losses)
    print("Mean OOF loss across folds", np.mean(oof_losses))
    print("STD OOF loss across folds", np.std(oof_losses))

    # set control train preds to 0
    control_mask = train["cp_type"] == "ctl_vehicle"
    oof_preds[control_mask] = 0

    print("OOF log loss: ", log_loss(np.ravel(y), np.ravel(oof_preds)))

    # create the submission file
    sub.iloc[:, 1:] = test_preds
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
