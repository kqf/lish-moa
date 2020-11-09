import numpy as np
import pandas as pd

from pathlib import Path
from functools import partial

from category_encoders import CountEncoder

from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

"""
Features:
    - "cp_type"
    - "cp_time"
    - "cp_dose" (D1, D2)
    - "g-*" (0-771)
    - "c-*" (0-99)
"""


class TypeConversion:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(np.float32)


class PandasSelector:
    def __init__(self, cols=None, startswith=None, exclude=None):
        self.cols = cols
        self.startswith = startswith
        self.exclude = exclude

    def fit(self, X, y=None):
        if self.cols is None and self.startswith is not None:
            self.cols = [c for c in X.columns if c.startswith(self.startswith)]
        return self

    def transform(self, X, y=None):
        if self.exclude is not None:
            return X.drop(columns=self.exclude)

        if self.cols is None:
            return X.to_numpy()

        return X[self.cols]


class GroupbyNormalizer:
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        gb = X.groupby(self.col)
        self.means = gb.mean()
        self.stds = gb.std()
        return self

    def transform(self, X, y=None):
        mu = pd.merge(X[self.col], self.means,
                      on=self.col).drop(columns=self.col)
        sigma = pd.merge(X[self.col], self.stds,
                         on=self.col).drop(columns=self.col)
        return (X.drop(columns=self.col) - mu) / sigma


class MeanEncoder:
    def __init__(self, col):
        self.col = col
        self.tcols = None

    def fit(self, X, y):
        self.tcols = [f"t-{i}" for i in range(y.shape[1])]

        df = pd.concat(
            [X[self.col], pd.DataFrame(y, columns=self.tcols)], axis=1)

        self.means = df.groupby(self.col)[self.tcols].mean()
        return self

    def transform(self, X):
        encoded = pd.merge(
            X[self.col],
            self.means,
            on=self.col,
            how="left"
        ).drop(columns=self.col)
        return encoded


def build_preprocessor():
    ce = make_pipeline(
        PandasSelector(),
        CountEncoder(
            cols=(0, 1, 2),
            return_df=False,
            min_group_size=1,  # Makes it possible to clone
        ),
        StandardScaler(),
        TypeConversion(),
    )

    c_quantiles = make_pipeline(
        PandasSelector(startswith="c-"),
        QuantileTransformer(n_quantiles=100, output_distribution="normal"),
        PCA(n_components=45),
    )

    g_quantiles = make_pipeline(
        PandasSelector(startswith="g-"),
        QuantileTransformer(n_quantiles=100, output_distribution="normal"),
        PCA(n_components=450),
    )

    pca_features = make_pipeline(
        make_union(
            c_quantiles,
            g_quantiles,
        ),
        VarianceThreshold(0.67),
        StandardScaler(),
    )

    ce = make_union(
        make_pipeline(
            PandasSelector(exclude=["cp_time", "cp_dose"]),
            MeanEncoder(["cp_type"]),
        ),
        make_pipeline(
            PandasSelector(exclude=["cp_type", "cp_dose"]),
            MeanEncoder(["cp_time"]),
        ),
        make_pipeline(
            PandasSelector(exclude=["cp_time", "cp_type"]),
            MeanEncoder(["cp_dose"]),
        ),
    )

    final = make_union(
        make_pipeline(
            ce,
            StandardScaler(),
        ),
        pca_features,
    )

    return final


def create_model(input_units, output_units, hidden_units=512, lr=1e-3):
    model = Sequential()
    model.add(
        Dense(hidden_units, activation="relu", input_shape=(input_units,))
    )
    model.add(Dense(output_units, activation="sigmoid"))
    model.compile(
        loss=["binary_crossentropy"],
        optimizer=Adam(
            lr=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=False,
            decay=0,
        )
    )
    return model


class DynamicKerasClassifier(KerasClassifier):
    def fit(self, X, y, **kwargs):
        self.build_fn = partial(
            self.build_fn,
            input_units=X.shape[1],
            output_units=y.shape[1]
        )

        # cut = 200. / X.shape[0]
        # freqs = y.mean(0)
        # self._freqs = freqs * (freqs < cut)
        return super().fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):
        probas = super().predict_proba(X, **kwargs)
        # NB: Average the labels
        # idx, = np.where(self._freqs > 0)
        # probas[:, idx] = (probas[:, idx] + self._freqs[idx]) / 2.
        return probas


def build_model():
    classifier = DynamicKerasClassifier(
        create_model,
        batch_size=128,
        epochs=5,
        validation_split=None,
        shuffle=True
    )

    model = make_pipeline(
        build_preprocessor(),
        classifier,
    )

    return model


def cv_fit(clf, X, y, X_test, cv=None, n_splits=5):
    cv = cv or KFold(n_splits=n_splits)

    test_preds = np.zeros((X_test.shape[0], y.shape[1]))

    losses_train = []
    losses_valid = []
    estimators = []
    for fn, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print("Starting fold: ", fn)

        estimators.append(clone(clf))
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # drop where cp_type==ctl_vehicle (baseline)
        ctl_mask = X_train.iloc[:, 0] == "ctl_vehicle"
        X_train = X_train[~ctl_mask]
        y_train = y_train[~ctl_mask]

        estimators[-1].fit(X_train, y_train)

        train_preds = estimators[-1].predict_proba(X_train)
        train_preds = np.nan_to_num(train_preds)  # positive class
        loss = log_loss(y_train.reshape(-1), train_preds.reshape(-1))
        losses_train.append(loss)

        val_preds = estimators[-1].predict_proba(X_val)
        val_preds = np.nan_to_num(val_preds)  # positive class
        loss = log_loss(y_val.reshape(-1), val_preds.reshape(-1))
        losses_valid.append(loss)

        preds = estimators[-1].predict_proba(X_test)
        preds = np.nan_to_num(preds)  # positive class
        test_preds += preds / cv.n_splits

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

    train_preds = clf.predict_proba(X_train)
    train_preds = np.nan_to_num(train_preds)  # positive class
    loss = log_loss(y_train.reshape(-1), train_preds.reshape(-1))
    losses_train.append(loss)
    losses_valid.append(loss)

    test_preds = clf.predict_proba(X_test)
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

    return df


def main():
    X = read_data("data/train_features.csv")
    y = read_data("data/train_targets_scored.csv")

    X_test = read_data("data/test_features.csv")
    sub = read_data("data/sample_submission.csv",
                    ignore_col=None, return_df=True)

    clf = build_model()
    clfs, losses_train, losses_valid, preds = cv_fit(clf, X, y, X_test)

    print(losses_valid)
    msg = "CV losses {} {:.4f} +/- {:.4f}"
    print(msg.format("train", losses_train.mean(), losses_train.std()))
    print(msg.format("valid", losses_valid.mean(), losses_valid.std()))

    # create the submission file
    sub.iloc[:, 1:] = preds
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
