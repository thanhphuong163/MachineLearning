import unittest
from utils.functions import my_square, my_pow


class TestFunctions(unittest.TestCase):
    def test_square_fn(self):
        n = 5
        msg = "Failed square function test: it should be 25."
        self.assertEqual(my_square(n), n*n, msg)

    def test_power_fn(self):
        self.assertEqual(my_pow(4, 3), 4**3,"Failed power function test: it should be 64.")

    def test_square_fn(self):
        self.assertEqual(my_square(4), 16, "Should be 16")
    
