import numpy as np
import pandas as pd

from functools import partial

from category_encoders import CountEncoder

from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class TypeConversion:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(np.float32)


def build_preprocessor():
    ce = make_pipeline(
        CountEncoder(
            cols=(0, 2),
            return_df=False,
            min_group_size=1,  # Makes it possible to clone
        ),
        StandardScaler(),
        TypeConversion(),
    )

    return ce


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
        return super().fit(X, y, **kwargs)


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
        X_train, X_val = X[trn_idx], X[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # drop where cp_type==ctl_vehicle (baseline)
        ctl_mask = X_train[:, 0] == "ctl_vehicle"
        X_train = X_train[~ctl_mask, :]
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
        test_preds += preds / n_splits

    return (
        estimators,
        np.array(losses_train),
        np.array(losses_valid),
        test_preds
    )


def read_data(path, ignore_col="sig_id", return_df=False):
    df = pd.read_csv(path)
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
    sub = read_data("data/sample_submission.csv", return_df=True)

    clf = build_model()
    clfs, losses_train, losses_valid, preds = cv_fit(clf, X, y, X_test)

    msg = "CV losses {} {:.4f} +/- {:.4f}"
    print(msg.format("train", losses_train.mean(), losses_train.std()))
    print(msg.format("valid", losses_valid.mean(), losses_valid.std()))

    # create the submission file
    sub.iloc[:, ] = preds
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
