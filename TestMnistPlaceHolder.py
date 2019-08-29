from unittest import TestCase
from MnistPlaceHolder import MnistPlaceHolder
import numpy as np
import tensorflow as tf
from numpy import dtype

class TestMnistPlaceHolder(TestCase):

    def test_construct(self):
        mnistPlaceHolder = MnistPlaceHolder()
        self.assertIsNotNone(mnistPlaceHolder, 'XXX')

    def test_load(self):
        mnistPlaceHolder = MnistPlaceHolder()
        (x_train, y_train), (x_test, y_test) = mnistPlaceHolder.load('')
        self.assertIsNotNone(x_train, msg='could not load mnist')
        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(x_train.dtype, np.uint8)

    def test_getEmbeddings(self):
        mnistPlaceHolder = MnistPlaceHolder()
        float32array = mnistPlaceHolder.getEmbeddings()
        self.assertTrue(float32array.dtype, np.float32)
