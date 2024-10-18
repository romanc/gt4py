# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import List

import dace
import dace.data
import dace.library
import dace.subsets

from gt4py.cartesian.gtc.dace import daceir as dcir


@dataclass
class SDFGContext:
    sdfg: dace.SDFG
    state: dace.SDFGState
    state_stack: List[dace.SDFGState] = field(default_factory=list)

    def add_state(self, *, label: str | None = None):
        new_state = self.sdfg.add_state(label)
        for edge in self.sdfg.out_edges(self.state):
            self.sdfg.remove_edge(edge)
            self.sdfg.add_edge(new_state, edge.dst, edge.data)
        self.sdfg.add_edge(self.state, new_state, dace.InterstateEdge())
        self.state = new_state
        return self

    def add_loop(self, index_range: dcir.Range):
        loop_state = self.sdfg.add_state()
        after_state = self.sdfg.add_state("loop_after")
        for edge in self.sdfg.out_edges(self.state):
            self.sdfg.remove_edge(edge)
            self.sdfg.add_edge(after_state, edge.dst, edge.data)

        assert isinstance(index_range.interval, dcir.DomainInterval)
        if index_range.stride < 0:
            initialize_expr = f"{index_range.interval.end} - 1"
            end_expr = f"{index_range.interval.start} - 1"
        else:
            initialize_expr = str(index_range.interval.start)
            end_expr = str(index_range.interval.end)
        comparison_op = "<" if index_range.stride > 0 else ">"
        condition_expr = f"{index_range.var} {comparison_op} {end_expr}"
        _, _, after_state = self.sdfg.add_loop(
            before_state=self.state,
            loop_state=loop_state,
            after_state=after_state,
            loop_var=index_range.var,
            initialize_expr=initialize_expr,
            condition_expr=condition_expr,
            increment_expr=f"{index_range.var}+({index_range.stride})",
        )
        if index_range.var not in self.sdfg.symbols:
            self.sdfg.add_symbol(index_range.var, stype=dace.int32)

        self.state_stack.append(after_state)
        self.state = loop_state
        return self

    def pop_loop(self):
        self._pop_last("loop_after")

    def add_condition(self):
        """Inserts a condition state after the current self.state.
        The condition state is connected to a true_state and a false_state based on
        a temporary local variable identified by `node.mask_name`. Both states then merge
        into a merge_state.
        self.state is set to true_state and merge_state / false_state are pushed to
        the stack of states; to be popped with `pop_condition_{false, after}()`.
        """
        merge_state = self.sdfg.add_state("condition_after")
        for edge in self.sdfg.out_edges(self.state):
            self.sdfg.remove_edge(edge)
            self.sdfg.add_edge(merge_state, edge.dst, edge.data)

        # evaluate node condition
        init_state = self.sdfg.add_state("condition_init")
        self.sdfg.add_edge(self.state, init_state, dace.InterstateEdge())

        # promote condition (from init_state) to symbol
        condition_state = self.sdfg.add_state("condition_guard")
        self.sdfg.add_edge(
            init_state,
            condition_state,
            # to be updated later (see usage of sdfg_ctx.add_condition())
            dace.InterstateEdge(assignments=dict(if_condition=None)),
        )

        true_state = self.sdfg.add_state("condition_true")
        self.sdfg.add_edge(
            condition_state, true_state, dace.InterstateEdge(condition="if_condition")
        )
        self.sdfg.add_edge(true_state, merge_state, dace.InterstateEdge())

        false_state = self.sdfg.add_state("condition_false")
        self.sdfg.add_edge(
            condition_state, false_state, dace.InterstateEdge(condition="not if_condition")
        )
        self.sdfg.add_edge(false_state, merge_state, dace.InterstateEdge())

        self.state_stack.append(merge_state)
        self.state_stack.append(false_state)
        self.state_stack.append(true_state)
        self.state_stack.append(condition_state)
        self.state = init_state
        return self

    def pop_condition_guard(self):
        self._pop_last("condition_guard")

    def pop_condition_true(self):
        self._pop_last("condition_true")

    def pop_condition_false(self):
        self._pop_last("condition_false")

    def pop_condition_after(self):
        self._pop_last("condition_after")

    def add_while(self):
        """Inserts a while loop after the current self.state.
        ...
        """
        after_state = self.sdfg.add_state("while_after")
        for edge in self.sdfg.out_edges(self.state):
            self.sdfg.remove_edge(edge)
            self.sdfg.add_edge(after_state, edge.dst, edge.data)

        # evaluate loop condition
        init_state = self.sdfg.add_state("while_init")
        self.sdfg.add_edge(self.state, init_state, dace.InterstateEdge())

        # promote condition (from init_state) to symbol
        guard_state = self.sdfg.add_state("while_guard")
        self.sdfg.add_edge(
            init_state,
            guard_state,
            # to be updated later (see usage of sdfg_ctx.add_while())
            dace.InterstateEdge(assignments=dict(loop_condition=None)),
        )

        loop_state = self.sdfg.add_state("while_loop")
        self.sdfg.add_edge(guard_state, loop_state, dace.InterstateEdge(condition="loop_condition"))
        # loop back to init_state to re-evaluate the loop condition
        self.sdfg.add_edge(loop_state, init_state, dace.InterstateEdge())

        # exit the loop
        self.sdfg.add_edge(
            guard_state, after_state, dace.InterstateEdge(condition="not loop_condition")
        )

        self.state_stack.append(after_state)
        self.state_stack.append(loop_state)
        self.state_stack.append(guard_state)
        self.state = init_state
        return self

    def pop_while_guard(self):
        self._pop_last("while_guard")

    def pop_while_loop(self):
        self._pop_last("while_loop")

    def pop_while_after(self):
        self._pop_last("while_after")

    def _pop_last(self, node_label: str | None = None) -> None:
        if node_label:
            assert self.state_stack[-1].label.startswith(node_label)

        self.state = self.state_stack[-1]
        del self.state_stack[-1]
