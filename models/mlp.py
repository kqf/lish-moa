import pandas as pd
import skorch
import torch

from category_encoders import CountEncoder
from sklearn.pipeline import make_pipeline
from skorch.toy import MLPModule


def build_model():
    classifier = skorch.NeuralNet(
        module=MLPModule,
        module__input_units=20,
        module__output_units=2,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.002,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=5,
        batch_size=128,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    model = make_pipeline(
        CountEncoder(cols=[0, 2], return_df=False),
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
    clf.fit(X, y)

    # create the submission file
    sub.iloc[:, 1:] = clf.predict_proba(X_test)
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
