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
"""Data structures."""


class DisjointSet:
  """A disjoint set, aka union-find data structure."""

  def __init__(self):
    self.parent = {}
    self.rank = {}

  def __repr__(self):
    return f'DisjointSet(parent={self.parent}, rank={self.rank})'

  def find(self, x):
    """Find the root of the set that x belongs to.

    If x is not in the set, insert it and return x.

    Args:
      x: the node to find the root of the set that it belongs to.

    Returns:
      The root of the set that x belongs to.
    """
    if x not in self.parent:
      self.parent[x] = x
      self.rank[x] = 0
      return x

    # Path compression.
    root = x
    while self.parent[root] != root:
      root = self.parent[root]
    while self.parent[x] != root:
      x, self.parent[x] = self.parent[x], root
    return root

  def union(self, x, y):
    """Merge the sets that x and y belong to."""
    x = self.find(x)
    y = self.find(y)
    if x == y:
      return
    # Ensure that x.rank >= y.rank.
    if self.rank[x] < self.rank[y]:
      x, y = y, x
    self.parent[y] = x
    if self.rank[x] == self.rank[y]:
      self.rank[x] += 1


class MergeableGraph:
  """An undirected graph where nodes can be merged."""

  def __init__(self):
    self._edges = set()  # a set of tuples, undirected
    self._nodes = DisjointSet()

  def add_edge(self, node1, node2):
    self._edges.add((self.get_root(node1), self.get_root(node2)))

  def merge_nodes(self, node1, node2):
    self._nodes.union(node1, node2)

  def get_root(self, node):
    return self._nodes.find(node)

  def get_edges(self):
    """Return all edges in the graph. The same edge will appear twice."""
    edges = set()
    for n1, n2 in self._edges:
      root1, root2 = self.get_root(n1), self.get_root(n2)
      if root1 != root2:
        edges.add((root1, root2))
        edges.add((root2, root1))
    return edges
