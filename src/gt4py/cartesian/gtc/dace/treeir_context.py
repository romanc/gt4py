# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Generic, TypeVar

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py.cartesian.gtc import definitions
from gt4py.cartesian.gtc.dace import treeir as tir


@dataclass
class TreeIRContext:
    root: tir.TreeRoot
    current_scope: tir.TreeScope

    field_extents: dict[str, definitions.Extent]  # field_name -> Extent
    block_extents: dict[int, definitions.Extent]  # id(horizontal execution) -> Extent


@dataclass
class ScheduleTreeContext:
    root: tn.ScheduleTreeRoot
    """A reference to the tree root."""

    current_scope: tn.ScheduleTreeScope
    """A reference to the current scope node."""


Context = TypeVar("Context", bound=TreeIRContext | ScheduleTreeContext)
Scope = TypeVar("Scope", bound=tir.TreeScope | tn.ScheduleTreeScope)


class ContextPushPop(Generic[Context, Scope]):
    """Append the node to the scope, then push/pop the scope."""

    def __init__(self, ctx: Context, node: Scope) -> None:
        self._ctx = ctx
        self._parent_scope = ctx.current_scope
        self._node = node

    def __enter__(self):
        self._node.parent = self._parent_scope
        self._parent_scope.children.append(self._node)
        self._ctx.current_scope = self._node

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._ctx.current_scope = self._parent_scope
