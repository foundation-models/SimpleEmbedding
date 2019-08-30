from unittest import TestCase
from MnistPlaceHolder import MnistPlaceHolder
import numpy as np
import tensorflow as tf
from numpy import dtype
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
        instance = MnistPlaceHolder()
        float32array = instance.get_embeddings()
        self.assertTrue(float32array.dtype, np.float32)

    def test_init(self):
        instance = MnistPlaceHolder()
        self.assertIsNotNone(instance.x_train, msg='could not load mnist')
        self.assertEqual(instance.x_train.shape, (60000, 28, 28))
        self.assertEqual(instance.x_train.dtype, np.uint8)
