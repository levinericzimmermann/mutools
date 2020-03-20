import collections
import unittest

from mutools import polyrhythms

from mu.rhy import binr


class PolyrhythmTest(unittest.TestCase):
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

    def test_Counter_difference(self) -> None:
        object0 = collections.Counter({2: 1, 3: 2, 5: 1})
        object1 = collections.Counter({2: 3, 3: 2})
        difference = collections.Counter({2: -2, 5: 1})
        function_result = polyrhythms.Polyrhythm.Counter_difference(object0, object1)

        self.assertEqual(function_result, difference)

    def test_find_polyrhythmic_identity(self) -> None:
        object0 = collections.Counter({2: 1, 3: 2, 5: 1})
        object1 = collections.Counter({2: 3})
        object2 = collections.Counter({2: 1, 5: 3})
        object3 = collections.Counter({7: 1, 11: 1})
        object4 = collections.Counter({5: 2, 11: 2})
        polyrhythmic_identity = collections.Counter({2: 3, 3: 2, 5: 3, 7: 1, 11: 2})

        function_result = polyrhythms.Polyrhythm.find_polyrhythmic_identity(
            (object0, object1, object2, object3, object4)
        )

        self.assertEqual(function_result, polyrhythmic_identity)

    def test_find_stretching_factor(self) -> None:
        polyrhythmic_identity = collections.Counter({2: 3, 3: 2, 5: 1, 7: 1})
        factorised_duration = collections.Counter({3: 2})

        expected_stretching_factor = (2 ** 3) * 5 * 7
        real_stretching_factor = polyrhythms.Polyrhythm.find_stretching_factor(
            factorised_duration, polyrhythmic_identity
        )
        self.assertEqual(expected_stretching_factor, real_stretching_factor)

    def test_transformed_rhythms(self) -> None:
        r00 = binr.Compound.from_euclid(8, 4)
        r01 = binr.Compound.from_euclid(8, 5)
        r02 = binr.Compound.from_euclid(8, 3)
        poly0 = polyrhythms.Polyrhythm(r00, r01, r02)

        # should be the same like the input since they all have the same size
        self.assertEqual(poly0.transformed_rhythms, (r00, r01, r02))

        r10 = binr.Compound.from_euclid(3, 2)
        r11 = binr.Compound.from_euclid(4, 2)
        poly1 = polyrhythms.Polyrhythm(r10, r11)

        # both should have the size of 12 (3 * 4 = 12)
        self.assertEqual(poly1.transformed_rhythms[0].beats, 12)
        self.assertEqual(poly1.transformed_rhythms[1].beats, 12)

        # both should be stretched version by their complementary size
        self.assertEqual(poly1.transformed_rhythms[0], r10.real_stretch(4))
        self.assertEqual(poly1.transformed_rhythms[1], r11.real_stretch(3))

        r20 = binr.Compound.from_euclid(3, 2)
        r21 = binr.Compound.from_euclid(4, 3)
        r22 = binr.Compound.from_euclid(5, 5)
        poly2 = polyrhythms.Polyrhythm(r20, r21, r22)

        # all of them should have the size of 60 (3 * 4 * 5 = 60)
        self.assertEqual(poly2.transformed_rhythms[0].beats, 60)
        self.assertEqual(poly2.transformed_rhythms[1].beats, 60)
        self.assertEqual(poly2.transformed_rhythms[2].beats, 60)

        # all of them should be a stretched version
        self.assertEqual(poly2.transformed_rhythms[0], r20.real_stretch(20))
        self.assertEqual(poly2.transformed_rhythms[1], r21.real_stretch(15))
        self.assertEqual(poly2.transformed_rhythms[2], r22.real_stretch(12))

        r30 = binr.Compound.from_euclid(6, 4)
        r31 = binr.Compound.from_euclid(4, 4)
        r32 = binr.Compound.from_euclid(5, 4)
        poly3 = polyrhythms.Polyrhythm(r30, r31, r32)

        # all of them should have the size of 60 (3 * 4 * 5 = 60)
        self.assertEqual(poly3.transformed_rhythms[0].beats, 60)
        self.assertEqual(poly3.transformed_rhythms[1].beats, 60)
        self.assertEqual(poly3.transformed_rhythms[2].beats, 60)

        # all of them should be a stretched version
        self.assertEqual(poly3.transformed_rhythms[0], r30.real_stretch(10))
        self.assertEqual(poly3.transformed_rhythms[1], r31.real_stretch(15))
        self.assertEqual(poly3.transformed_rhythms[2], r32.real_stretch(12))
