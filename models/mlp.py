import torch
import skorch
import numpy as np
import pandas as pd

from skorch.toy import MLPModule
from category_encoders import CountEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ConvertTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.to_numpy()


class TypeConversion:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(np.float32)


def build_preprocessor():
    ce = make_pipeline(
        ConvertTransformer(),
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
        net.set_params(module__input_units=X.shape[1])
        net.set_params(module__output_units=y.shape[1])

        n_pars = self.count_parameters(net.module_)
        print(f'The train data is of {X.shape} shape')
        print(f'The train labels are of {y.shape} shape')
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model():
    classifier = skorch.NeuralNet(
        module=MLPModule,
        module__input_units=875,
        module__output_units=206,
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


def read_data(path, ignore_col="sig_id"):
    df = pd.read_csv(path)
    df.drop(columns=[ignore_col], inplace=True)
    return df


def main():
    X = read_data("data/train_features.csv")
    y = read_data("data/train_targets_scored.csv")

    X_test = read_data("data/test_features.csv")
    sub = read_data("data/sample_submission.csv")

    clf = build_model()
    clf.fit(X, y.to_numpy().astype(np.float32))

    # create the submission file
    sub.iloc[:, ] = clf.predict_proba(X_test)[:, 1, :]
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
