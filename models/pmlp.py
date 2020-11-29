import numpy as np
import pymc3 as pm
import theano

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from warnings import filterwarnings
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))

floatX = theano.config.floatX
filterwarnings('ignore')


def construct_nn(X, y, hidden_units=5):
    nh = hidden_units

    # Initialize random weights between each layer
    ifc1 = np.random.randn(X.shape[1], nh).astype(floatX)
    ifc2 = np.random.randn(nh, nh).astype(floatX)
    ifc3 = np.random.randn(nh).astype(floatX)

    with pm.Model() as model:
        """
            Trick: Turn inputs and outputs into shared variables using the data container pm.Data
            It's still the same thing, but we can later change the values of the shared variable
            (to switch in the test-data later) and pymc3 will just use the new data.
            Kind-of like a pointer we can redirect.
            For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        """  # noqa
        ann_input = pm.Data('ann_input', X)
        ann_output = pm.Data('ann_output', y)

        # Weights from input to hidden layer
        f1 = pm.Normal('f1', 0, sigma=1, shape=(X.shape[1], nh), testval=ifc1)

        # Weights from 1st to 2nd layer
        fc2 = pm.Normal('fc2', 0, sigma=1, shape=(nh, nh), testval=ifc2)

        # Weights from hidden layer to output
        fc3 = pm.Normal('fc3', 0, sigma=1, shape=(nh,), testval=ifc3)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, f1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, fc2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, fc3))

        # Binary classification -> Bernoulli likelihood
        pm.Bernoulli('out',
                     act_out,
                     observed=ann_output,
                     # IMPORTANT for minibatches
                     total_size=y.shape[0]
                     )
    return model


def main():
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X = scale(X)
    X = X.astype(floatX)
    Y = Y.astype(floatX)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

    model = construct_nn(X_train, Y_train)
    with model:
        inference = pm.ADVI()
        approx = pm.fit(n=30000, method=inference)


if __name__ == '__main__':
    main()
