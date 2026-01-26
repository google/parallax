# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from parallax.sharding import data_structures
from absl.testing import absltest


class DataStructuresTest(absltest.TestCase):

  def test_disjoint_set(self):
    s = data_structures.DisjointSet()
    s.union(1, 2)
    s.union(2, 3)
    s.union(3, 4)
    s.union(4, 5)
    s.union(6, 7)
    self.assertEqual(s.find(1), s.find(5))
    self.assertNotEqual(s.find(1), s.find(7))

  def test_disjoint_set_union_by_rank(self):
    def get_depth(s, x):
      depth = 0
      while s.parent[x] != x:
        depth += 1
        x = s.parent[x]
      return depth

    s1 = data_structures.DisjointSet()
    s2 = data_structures.DisjointSet()
    for i in range(100):
      s1.union(i, i + 1)
      s2.union(i + 1, i)

    for i in range(100):
      self.assertLess(get_depth(s1, i), 10)
      self.assertLess(get_depth(s2, i), 10)

  def test_mergeable_graph(self):
    g = data_structures.MergeableGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(3, 4)

    # Test merged nodes.
    self.assertNotEqual(g.get_root(2), g.get_root(3))
    g.merge_nodes(2, 3)
    self.assertEqual(g.get_root(2), g.get_root(3))

    # Test unmerged nodes.
    self.assertNotEqual(g.get_root(1), g.get_root(2))
    self.assertNotEqual(g.get_root(1), g.get_root(3))
    self.assertNotEqual(g.get_root(3), g.get_root(4))

    self.assertLen(g.get_edges(), 4)


if __name__ == "__main__":
  absltest.main()
