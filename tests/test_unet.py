import sys
import unittest

class TestUnet(unittest.TestCase):

    def test_something(self):
        self.assertAlmostEqual(3, 3.3)

    def test_something1(self):
        self.assertCountEqual([3, 3, 3], [3, 3, 3])

if __name__ == '__main__':
    unittest.main()