import numpy as np
import pymc3 as pm
import theano
from pymc3.theanof import set_tt_rng, MRG_RandomStreams

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import NotFittedError

from warnings import filterwarnings

set_tt_rng(MRG_RandomStreams(42))
floatX = theano.config.floatX
theano.config.compute_test_value = 'off'
filterwarnings('ignore')


def construct_nn(X, y, hidden_units=512):
    nh = hidden_units

    # Initialize random weights between each layer
    ifc1 = np.random.randn(X.shape[1], nh).astype(floatX)
    ifc2 = np.random.randn(nh, nh).astype(floatX)
    ifc3 = np.random.randn(nh, y.shape[1]).astype(floatX)

    with pm.Model() as model:
        """
            Trick: Turn inputs and outputs into shared variables using the data container pm.Data
            It's still the same thing, but we can later change the values of the shared variable
            (to switch in the test-data later) and pymc3 will just use the new data.
            Kind-of like a pointer we can redirect.
            For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        """  # noqa
        _input = pm.Data('_input', X)
        _output = pm.Data('_output', y)

        # Weights from input to hidden layer
        f1 = pm.Normal('f1', 0, sigma=1, shape=(X.shape[1], nh), testval=ifc1)

        # Weights from 1st to 2nd layer
        fc2 = pm.Normal('fc2', 0, sigma=1, shape=(nh, nh), testval=ifc2)

        # Weights from hidden layer to output
        fc3 = pm.Normal('fc3', 0, sigma=1,
                        shape=(nh, y.shape[1]), testval=ifc3)

        # Build neural-network using tanh activation function
        act_1 = theano.tensor.nnet.relu(pm.math.dot(_input, f1))
        act_2 = theano.tensor.nnet.relu(pm.math.dot(act_1, fc2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, fc3))

        # Binary classification -> Bernoulli likelihood
        pm.Categorical(
            'out',
            act_out,
            observed=_output.T,
            # IMPORTANT for minibatches
            total_size=y.shape[0]
        )

    return model


def create_inference(approx, model):
    # create symbolic input
    x = theano.tensor.matrix('X')

    # symbolic number of samples is supported,
    # we build vectorized posterior on the fly
    n = theano.tensor.iscalar('n')

    # Do not forget test_values or set
    _sample_proba = approx.sample_node(
        model.out.distribution.p,
        size=n,
        more_replacements={model['_input']: x})
    # It is time to compile the function
    # No updates are needed for Approximation random generator
    # Efficient vectorized form of sampling is
    return theano.function([x, n], _sample_proba)


class BayesianClassifer:
    def __init__(self, build_model, inf_samples=512, n=50_000):
        self.build_model = build_model
        self.model = None
        self.sample = None
        self.n = n
        self.inf_samples = inf_samples

    def fit(self, X, y):
        self.model = self.build_model(X, y)
        with self.model:
            self.approx = pm.fit(n=self.n, method=pm.ADVI())
            self.sample = create_inference(self.approx, self.model)

    def predict_proba(self, X):
        if self.sample is None:
            raise NotFittedError("Please call model.fit(X, y) first")
        samples = self.sample(X, self.inf_samples)
        # Average over inf_samples dimension
        return samples.mean(0)

    def predict(self, X):
        return self.predict_proba(X) > 0.5


def main():
    # X, y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X, y = make_multilabel_classification(random_state=0, n_samples=1000)
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    model = BayesianClassifer(build_model=construct_nn, n=1000)
    model.fit(X_train, y_train)

    print("Train accuracy", accuracy_score(model.predict(X_train), y_train))
    print("Valid accuracy", accuracy_score(model.predict(X_test), y_test))


if __name__ == '__main__':
    main()