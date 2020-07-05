import unittest

from mutools import alternating_hands


class FunctionTests(unittest.TestCase):
    """Test differnt functions."""

    """
    def test_mirror(self) -> None:
        argument0 = (True, False, False, True, False)
        expected0 = (False, True, True, False, True)

        argument1 = (False, False, True, True, False)
        expected1 = (True, True, False, False, True)

        self.assertEqual(alternating_hands.__mirror(argument0), expected0)
        self.assertEqual(alternating_hands.__mirror(argument1), expected1)
    """

    def test_paradiddle(self) -> None:
        argument0 = 4
        expected0 = (((0, 2, 3, 5), (1, 4, 6, 7)), 8)

        self.assertEqual(alternating_hands.paradiddle(argument0), expected0)
