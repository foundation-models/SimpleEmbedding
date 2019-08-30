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

    def test_regression_init(self):
        instance = Regression(2)
        self.assertTrue(np.array_equal([[0.],[0.]], instance.w))

    def test_sigmoid(self):
        instance = Regression()
        self.assertEqual(0.5, instance.sigmoid(0))