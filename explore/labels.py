import numpy as np
import matplotlib.pyplot as plt


from models.mlp import read_data


def plotable(a):
    a = a.astype(float)
    a[a == 0] = np.nan
    return a


def main():
    y = read_data("data/train_targets_scored.csv", return_df=True)
    coo = y.T @ y

    # Check conditional probabilities
    interaction = coo.values.copy() / coo.values.sum(axis=1, keepdims=True)
    np.fill_diagonal(interaction, 0)
    nonzero = ~np.all(interaction == 0, axis=0)
    plt.matshow(plotable(interaction[np.ix_(nonzero, nonzero)]))
    plt.show()

    plt.matshow(plotable(coo.values))
    plt.show()

    # plt.yticks(np.arange(coo.shape[0]), y.columns)

if __name__ == '__main__':
    main()
