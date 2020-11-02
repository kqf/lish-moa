import numpy as np
import pandas as pd

from pathlib import Path


from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class ConstantClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.probas_ = None

    def fit(self, X, y):
        if np.any((y != 0) & (y != 1)):
            raise IOError("ConstantClassifier suports only binary labels")

        self.probas_ = y.mean(axis=0)
        return self

    def predict(self, X):
        return np.vstack([self.probas_] * X.shape[0])


def build_model():
    return ConstantClassifier()


def cv_fit(clf, X, y, X_test, cv=None, n_splits=5):
    cv = cv or KFold(n_splits=n_splits)

    test_preds = np.zeros((X_test.shape[0], y.shape[1]))

    losses_train = []
    losses_valid = []
    estimators = []
    for fn, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print("Starting fold: ", fn)

        estimators.append(clone(clf))
        X_train, X_val = X[trn_idx], X[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # drop where cp_type==ctl_vehicle (baseline)
        ctl_mask = X_train[:, 0] == "ctl_vehicle"
        X_train = X_train[~ctl_mask, :]
        y_train = y_train[~ctl_mask]

        estimators[-1].fit(X_train, y_train)

        train_preds = estimators[-1].predict(X_train)
        train_preds = np.nan_to_num(train_preds)  # positive class
        loss = log_loss(y_train.reshape(-1), train_preds.reshape(-1))
        losses_train.append(loss)

        val_preds = estimators[-1].predict(X_val)
        val_preds = np.nan_to_num(val_preds)  # positive class
        loss = log_loss(y_val.reshape(-1), val_preds.reshape(-1))
        losses_valid.append(loss)

        preds = estimators[-1].predict(X_test)
        preds = np.nan_to_num(preds)  # positive class
        test_preds += preds / n_splits

    return (
        estimators,
        np.array(losses_train),
        np.array(losses_valid),
        test_preds
    )


def fit(clf, X, y, X_test):
    losses_train = []
    losses_valid = []
    ctl_mask = X[:, 0] == "ctl_vehicle"
    X_train = X[~ctl_mask, :]
    y_train = y[~ctl_mask]
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    train_preds = np.nan_to_num(train_preds)  # positive class
    loss = log_loss(y_train.reshape(-1), train_preds.reshape(-1))
    losses_train.append(loss)
    losses_valid.append(loss)

    test_preds = clf.predict(X_test)
    test_preds = np.nan_to_num(test_preds)  # positive class
    return (
        clf,
        np.array(losses_train),
        np.array(losses_valid),
        test_preds
    )


def read_data(path, ignore_col="sig_id", return_df=False):
    file_path = Path(path)
    if not file_path.is_file():
        file_path = Path("/kaggle/input/lish-moa/") / file_path.name

    df = pd.read_csv(file_path)
    if ignore_col is not None:
        df.drop(columns=[ignore_col], inplace=True)

    if return_df:
        return df

    if df.shape[1] == 206:
        return df.to_numpy().astype(np.float32)

    return df.to_numpy()


def main():
    X = read_data("data/train_features.csv")
    y = read_data("data/train_targets_scored.csv")

    X_test = read_data("data/test_features.csv")
    sub = read_data("data/sample_submission.csv",
                    ignore_col=None, return_df=True)

    clf = build_model()
    # clfs, losses_train, losses_valid, preds = cv_fit(clf, X, y, X_test)
    clfs, losses_train, losses_valid, preds = fit(clf, X, y, X_test)

    msg = "CV losses {} {:.4f} +/- {:.4f}"
    print(msg.format("train", losses_train.mean(), losses_train.std()))
    print(msg.format("valid", losses_valid.mean(), losses_valid.std()))

    # create the submission file
    sub.iloc[:, 1:] = preds
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
