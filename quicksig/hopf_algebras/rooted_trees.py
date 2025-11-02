"""Utilities for working with rooted trees and forests.

This module defines the ``Forest`` container type and helpers to render
collections of rooted trees as Unicode art. A forest is represented by a
"parent array" for each tree, stacked along the first axis.

Conventions
- Nodes are indexed in preorder with the root at index 0.
- For each tree, ``parent[0] == -1`` and for all ``i > 0``, ``0 <= parent[i] < i``.
"""

from typing import NamedTuple

import jax.numpy as jnp


class Forest(NamedTuple):
    """A batch container for a forest of rooted trees.

    Parameters
    - parent: 2D array of shape ``(num_trees, n)`` with dtype ``int32``.
      Each row encodes one rooted tree via its parent array in preorder:
      ``parent[0] == -1`` and for ``i > 0`` we have ``0 <= parent[i] < i``.

    Notes
    - This container is compatible with JAX; the array can be a ``jax.Array``.
    - The number of nodes ``n`` is the same for all trees in the forest.

    Example
    >>> import jax.numpy as jnp
    >>> forest = Forest(parent=jnp.array([[-1, 0, 0]], dtype=jnp.int32))
    >>> forest.parent.shape
    (1, 3)
    """

    parent: jnp.ndarray


def _build_children(parent: list[int]) -> list[list[int]]:
    """Compute adjacency lists (children) from a parent array.

    Args:
        parent: A single-tree parent array in preorder, length ``n``.

    Returns:
        A list of length ``n`` where entry ``i`` contains the child indices of ``i``.

    Complexity:
        O(n) time and O(n) additional space.
    """
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        p = parent[i]
        if p >= 0:
            children[p].append(i)
    return children


def _render_tree_unicode(parent: list[int], show_ids: bool) -> list[str]:
    """Render a single rooted tree to Unicode lines.

    Args:
        parent: Parent array for one tree (preorder indexing, root at 0).
        show_ids: If ``True``, append node indices to bullets (e.g., ``•5``).

    Returns:
        A list of strings, each a line of the rendered tree using box-drawing
        characters.
    """
    # Rooted at node 0; draw using box-drawing characters
    children = _build_children(parent)

    def node_label(i: int) -> str:
        return f"•{i}" if show_ids else "•"

    lines: list[str] = []

    def dfs(node: int, prefix: str, is_last: bool) -> None:
        if node == 0:
            lines.append(node_label(node))
        else:
            branch = "└─ " if is_last else "├─ "
            lines.append(prefix + branch + node_label(node))
        # Root's immediate children should not be indented; deeper levels follow the usual rules.
        if node == 0:
            next_prefix = ""
        else:
            next_prefix = prefix + ("   " if is_last else "│  ")
        for idx, child in enumerate(children[node]):
            dfs(child, next_prefix, idx == len(children[node]) - 1)

    dfs(0, "", True)
    return lines


def print_forest(batch: Forest, show_node_ids: bool = False) -> str:
    """Render a ``Forest`` as a fenced Markdown code block.

    Args:
        batch: A ``Forest`` with ``parent`` of shape ``(num_trees, n)``.
        show_node_ids: If ``True``, include node indices next to bullets.

    Returns:
        A single string containing a fenced code block with one Unicode tree
        per forest row, separated by a blank line.
    """
    parents = jnp.asarray(batch.parent)
    drawings: list[str] = []
    for row in range(parents.shape[0]):
        parent_row: list[int] = list(map(int, parents[row].tolist()))
        tree_lines = _render_tree_unicode(parent_row, show_node_ids)
        drawings.append("\n".join(tree_lines))
    body = "\n\n".join(drawings)
    return f"```\n{body}\n```"


if __name__ == "__main__":
    from quicksig.hopf_algebras.bck_trees import enumerate_bck_trees
    from quicksig.hopf_algebras.mkw_trees import enumerate_mkw_trees

    batch_bck = enumerate_bck_trees(5)
    print(print_forest(batch_bck))

    batch_mkw = enumerate_mkw_trees(5)
    print(print_forest(batch_mkw))
