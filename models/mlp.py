import torch
import skorch
import numpy as np
import pandas as pd

from skorch.toy import MLPModule
from category_encoders import CountEncoder

from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


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


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(
            module__input_units=X.shape[1],
            module__output_units=y.shape[1]
        )

        n_pars = self.count_parameters(net.module_)
        print(f'The train data is of {X.shape} shape')
        print(f'The train labels are of {y.shape} shape')
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        train_preds = np.nan_to_num(train_preds[:, 1, :])  # positive class
        loss = log_loss(y_train.reshape(-1), train_preds.reshape(-1))
        losses_train.append(loss)

        val_preds = estimators[-1].predict_proba(X_val)
        val_preds = np.nan_to_num(val_preds[:, 1, :])  # positive class
        loss = log_loss(y_val.reshape(-1), val_preds.reshape(-1))
        losses_valid.append(loss)

        preds = estimators[-1].predict_proba(X_test)
        preds = np.nan_to_num(preds[:, 1, :])  # positive class
        test_preds += preds / n_splits

    return (
        estimators,
        np.array(losses_train),
        np.array(losses_valid),
        test_preds
    )


def build_model():
    classifier = skorch.NeuralNet(
        module=MLPModule,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.BCEWithLogitsLoss,
        max_epochs=5,
        batch_size=128,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        train_split=None,
        callbacks=[
            DynamicVariablesSetter(),
        ],
    )

    model = make_pipeline(
        build_preprocessor(),
        classifier,
    )

    return model


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
