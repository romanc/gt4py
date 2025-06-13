# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TypeAlias

from dace import Memlet, data, dtypes, nodes

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import daceir as dcir, treeir_context as tctx


SymbolDict: TypeAlias = dict[str, dtypes.typeclass]


class Bounds(eve.Node):
    start: str
    end: str


class TreeNode(eve.Node):
    parent: TreeScope | None


class TreeScope(TreeNode):
    children: list[TreeScope | TreeNode]

    def open(self, ctx: tctx.TreeIRContext) -> tctx.ContextPushPop[tctx.TreeIRContext, TreeScope]:
        return tctx.ContextPushPop[tctx.TreeIRContext, TreeScope](ctx, self)


class Tasklet(TreeNode):
    tasklet: nodes.Tasklet

    inputs: dict[str, Memlet]
    """Mapping tasklet.in_connectors to Memlets"""
    outputs: dict[str, Memlet]
    """Mapping tasklet.out_connectors to Memlets"""


class IfElse(TreeScope):
    # This should become an if/else, someday, so I am naming it if/else in hope
    # to see it before my bodily demise
    if_condition_code: str
    """Condition as ScheduleTree worthy code"""


class While(TreeScope):
    condition_code: str
    """Condition as ScheduleTree worthy code"""


class HorizontalLoop(TreeScope):
    bounds_i: Bounds
    bounds_j: Bounds


class VerticalLoop(TreeScope):
    loop_order: common.LoopOrder
    bounds_k: Bounds


class TreeRoot(TreeScope):
    name: str

    containers: dict[str, data.Data]
    """Mapping field/scalar names to data descriptors."""

    dimensions: dict[str, tuple[bool, bool, bool]]
    """Mapping field names to shape-axis."""

    shift: dict[str, dict[dcir.Axis, int]]
    """Mapping field names to dict[axis] -> shift."""

    symbols: SymbolDict
    """Mapping between type and symbol name."""
