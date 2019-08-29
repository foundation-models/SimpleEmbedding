from unittest import TestCase
from MnistPlaceHolder import MnistPlaceHolder
import numpy as np
import tensorflow as tf
from numpy import dtype

class TestMnistPlaceHolder(TestCase):

    def test_construct(self):
        myclass = MnistPlaceHolder()
        self.assertIsNotNone(myclass, 'XXX')

    def test_load(self):
        myclass = MnistPlaceHolder()
        (x_train, y_train), (x_test, y_test) = myclass.load('')
        self.assertIsNotNone(x_train, msg='could not load mnist')
        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(x_train.dtype, np.uint8)

    def test_getEmbeddings(self):
        myclass = MnistPlaceHolder()
        float32array = myclass.getEmbeddings()
        self.assertTrue(float32array.dtype, np.float32)
        
