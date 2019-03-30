from unittest import TestCase

from graph_nets.data_structures.attribute import Attribute
from graph_nets.data_structures.node import Node
from graph_nets.data_structures.utils import sets_equal, lists_equal


class TestUtils(TestCase):

    def test_sets_equal_ordering(self):
        s1 = {1, 2, 3, 4}
        s2 = {2, 3, 4, 1}
        self.assertTrue(sets_equal(s1, s2))

    def test_sets_equal_objects_comparator(self):
        s1 = {Node(Attribute(1)), Node(Attribute(2)), Node(Attribute("test"))}
        s2 = {Node(Attribute(2)), Node(Attribute(1)), Node(Attribute("test"))}
        self.assertTrue(sets_equal(s1, s2, comparator=Node.eq_attr))

    def test_sets_different_length(self):
        s1 = {1, 2}
        s2 = {}
        self.assertFalse(sets_equal(s1, s2))

    def test_lists_equal(self):
        l1 = [1, 2]
        l2 = [1, 2]
        self.assertTrue(lists_equal(l1, l2))

        l1.append(3)
        l2.append(4)
        self.assertFalse(lists_equal(l1, l2))

    def test_lists_equal_objects_comparator(self):
        l1 = [Node(Attribute(1)), Node(Attribute(2)), Node(Attribute("test"))]
        l2 = [Node(Attribute(1)), Node(Attribute(2)), Node(Attribute("test"))]
        self.assertTrue(lists_equal(l1, l2, comparator=Node.eq_attr))
