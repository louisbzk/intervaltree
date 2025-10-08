"""
intervaltree: A mutable, self-balancing interval tree for Python 2 and 3.
Queries may be by point, by range overlap, or by range envelopment.

Test module: IntervalTree, Special methods

Copyright 2013-2018 Chaim Leib Halbert

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

################################################################################
# Louis Blazejczak 2025 ########################################################
# Test "range-like" intervaltree methods #######################################
################################################################################

import pytest

from intervaltree import Interval, IntervalTree


@pytest.mark.parametrize(
    "itree, strict, expected",
    [
        (
            IntervalTree([Interval(0, 1), Interval(1, 2), Interval(2.0000001, 3)]),
            True,
            IntervalTree([Interval(0, 1), Interval(1, 2), Interval(2.0000001, 3)]),
        ),
        (
            IntervalTree([Interval(0, 1), Interval(1, 2), Interval(2.0000001, 3)]),
            False,
            IntervalTree([Interval(0, 2), Interval(2.0000001, 3)]),
        ),
        (
            IntervalTree(
                [
                    Interval(0, 1, "foo"),
                    Interval(1, 2, "foo"),
                    Interval(2, 3, "bar"),
                    Interval(3, 4, "foo"),
                ]
            ),
            False,
            IntervalTree(
                [Interval(0, 2, "foo"), Interval(2, 3, "bar"), Interval(3, 4, "foo")]
            ),
        ),
        (
            IntervalTree(
                [
                    Interval(0, 1, "foo"),
                    Interval(1, 2, "foo"),
                    Interval(2, 3, "bar"),
                    Interval(3, 4, "foo"),
                ]
            ),
            True,
            IntervalTree(
                [
                    Interval(0, 1, "foo"),
                    Interval(1, 2, "foo"),
                    Interval(2, 3, "bar"),
                    Interval(3, 4, "foo"),
                ]
            ),
        ),
    ],
)
def test_merge_overlaps_strict_data(
    itree: IntervalTree,
    strict,
    expected,
):
    # the cases where strict_data=False are already tested in other files
    result = itree.merge_overlaps(
        strict=strict,
        strict_data=True,
    )
    assert result == expected


@pytest.mark.parametrize(
    "tree1, tree2, data_intersect_fn, expected",
    [
        (
            IntervalTree([Interval(0, 5, "foo"), Interval(6, 10)]),
            IntervalTree([Interval(1, 3), Interval(4, 7, "bar"), Interval(7, 15)]),
            None,
            IntervalTree(
                [Interval(1, 3), Interval(4, 5), Interval(6, 7), Interval(7, 10)]
            ),
        ),
        (
            IntervalTree([Interval(0, 5, "foo"), Interval(6, 10)]),
            IntervalTree([Interval(1, 3), Interval(4, 7, "bar"), Interval(7, 15)]),
            lambda x1, x2: "_".join([str(x1), str(x2)]),
            IntervalTree(
                [
                    Interval(1, 3, "foo_None"),
                    Interval(4, 5, "foo_bar"),
                    Interval(6, 7, "None_bar"),
                    Interval(7, 10, "None_None"),
                ]
            ),
        ),
    ],
)
def test_range_intersect(
    tree1: IntervalTree,
    tree2: IntervalTree,
    data_intersect_fn,
    expected,
):
    result = tree1.range_intersection(tree2, data_intersect_fn=data_intersect_fn)
    assert result == expected


def test_range_contains():
    tree = IntervalTree([Interval(0, 5, "foo"), Interval(6, 10)])
    assert tree.range_contains(Interval(0, 5, "foo"))
    assert tree.range_contains(Interval(0, 5))
    assert tree.range_contains(Interval(1, 4))
    assert tree.range_contains(Interval(1, 4, "bar"))
    assert tree.range_contains(Interval(7, 9, "bar"))
    assert not tree.range_contains(Interval(1, 8))
    assert not tree.range_contains(Interval(-1, 4))
    assert not tree.range_contains(Interval(-1, 5.5))
    assert not tree.range_contains(Interval(5.1, 5.9))
    assert not tree.range_contains(Interval(8, 11))


# -------------------
# Tests for itree_invert
# -------------------


def test_itree_invert_simple():
    tree = IntervalTree([Interval(2, 5)])
    inverted = tree.invert(min_value=0, max_value=10)
    expected = IntervalTree([Interval(0, 2), Interval(5, 10)])
    assert set(inverted) == set(expected)


def test_itree_invert_empty_tree():
    tree = IntervalTree()
    inverted = tree.invert(min_value=0, max_value=5)
    expected = IntervalTree([Interval(0, 5)])
    assert set(inverted) == set(expected)


def test_itree_invert_unbounded_range():
    tree = IntervalTree([Interval(1, 3)])
    inverted = tree.invert()
    intervals = sorted(inverted, key=lambda i: i.begin)
    assert intervals[0].begin == -float("inf")
    assert intervals[0].end == 1
    assert intervals[1].begin == 3
    assert intervals[1].end == float("inf")


def test_itree_invert_with_overlaps():
    tree = IntervalTree([Interval(1, 3), Interval(2, 4)])
    with pytest.raises(ValueError, match="tree has overlaps in the inversion range"):
        tree.invert()


def test_itree_invert_with_clipping():
    tree = IntervalTree([Interval(0, 10)])
    inverted = tree.invert(min_value=2, max_value=8)
    assert inverted == IntervalTree([])
