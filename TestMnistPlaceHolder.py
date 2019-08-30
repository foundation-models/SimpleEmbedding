from unittest import TestCase
from MnistPlaceHolder import MnistWrapper
from MnistPlaceHolder import Regression
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import warnings


def fxn() -> object:
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings(record=True) as w:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("always")
    # Trigger a warning.
    fxn()
    # Verify some things
    assert len(w) == 1
    assert issubclass(w[-1].category, DeprecationWarning)
    assert "deprecated" in str(w[-1].message)


class TestMnistPlaceHolder(TestCase):

    def test_get_embeddings(self):
        instance = MnistWrapper()
        embeddings = instance.get_embeddings()
        plt.plot(embeddings[0])
        plt.show()
        self.assertTrue(embeddings.dtype, np.float32)

    def test_mnist_init(self):
        instance = MnistWrapper()
        #plt.imshow(instance.x_train[0])
        #plt.show()
        self.assertIsNotNone(instance.x_train, msg='could not load mnist')
        self.assertEqual(instance.x_train.shape, (60000, 28, 28))
        self.assertEqual(max(instance.x_train[0].reshape(-1)), 255)
        self.assertEqual(instance.x_train.dtype, np.uint8)

    def test_sigmoid(self):
        self.assertEqual(0.5, Regression.sigmoid(0))

    def test_propagate(self):
        w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
        instance = Regression(X,Y)
        grads, cost = instance.propagate(w,b)
        self.assertTrue(np.allclose([[0.99993216],[1.99980262]],grads["dw"]))
        self.assertTrue(np.allclose(0.49993523062470574, grads["db"]))
        self.assertTrue(np.allclose(6.000064773192205, cost))

