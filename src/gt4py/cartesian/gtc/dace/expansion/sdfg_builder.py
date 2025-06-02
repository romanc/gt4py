# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ChainMap, Dict, List, Optional, Set

import dace
from dace import nodes

from gt4py import eve
from gt4py.cartesian.gtc.dace import daceir as dcir, prefix
from gt4py.cartesian.gtc.dace.expansion.tasklet_codegen import TaskletCodegen
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gt4py.cartesian.gtc.dace.utils import get_dace_debuginfo


def _make_dace_range(
    access_info: dcir.FieldAccessInfo, data_dims: tuple[int, ...]
) -> dace.subsets.Range:
    ranges = []

    # cartesian dimensions
    for axis in access_info.axes():
        subset_start, subset_end = access_info.grid_subset.intervals[axis].to_dace_symbolic()
        ranges.append((subset_start, subset_end - 1, 1))  # TODO: always stride 1?

    # data dimensions
    ranges.extend((0, dim - 1, 1) for dim in data_dims)
    return dace.subsets.Range(ranges)


def _make_dace_memlet(
    memlet: dcir.Memlet, symtable: ChainMap[eve.SymbolRef, dcir.Decl]
) -> dace.Memlet:
    field_decl = symtable[memlet.field]
    assert isinstance(field_decl, dcir.FieldDecl)
    return dace.Memlet(
        memlet.field,
        subset=_make_dace_range(memlet.access_info, field_decl.data_dims),
        dynamic=field_decl.is_dynamic,
    )


class StencilComputationSDFGBuilder(eve.VisitorWithSymbolTableTrait):
    @dataclass
    class NestedSDFGContext:
        reads: set[eve.SymbolRef]
        writes: set[eve.SymbolRef]

    @dataclass
    class DomainMapContext:
        map_entry: nodes.MapEntry
        map_exit: nodes.MapExit
        # per scope access_cache dict[field_name] = (AccessNode, None) or (MapEntry, connector)
        access_cache: dict[str, tuple[nodes.AccessNode, None] | tuple[nodes.MapEntry, str]]
        # book-keeping in case the closing MapExit isn't connect to anything yet
        last_child: nodes.Node | None

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        state_stack: List[dace.SDFGState] = dataclasses.field(default_factory=list)

        # per state access_node cache
        access_cache: dict[dace.SDFGState, dict[eve.SymbolRef, nodes.AccessNode]] = (
            dataclasses.field(default_factory=dict)
        )

        def add_state(self, label: Optional[str] = None) -> None:
            new_state = self.sdfg.add_state(label=label)
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(new_state, edge.dst, edge.data)
            self.sdfg.add_edge(self.state, new_state, dace.InterstateEdge())
            self.state = new_state

        def add_loop(self, index_range: dcir.Range) -> None:
            loop_state = self.sdfg.add_state("loop_state")
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

        def pop_loop(self) -> None:
            self._pop_last("loop_after")

        def add_condition(self, node: dcir.Condition) -> None:
            """Inserts a condition after the current self.state.

            The condition consists of an initial state connected to a guard state, which branches
            to a true_state and a false_state based on the given condition. Both states then merge
            into a merge_state.

            self.state is set to init_state and the other states are pushed on the stack to be
            popped with `pop_condition_*()` methods.
            """
            # Data model validators enforce this to exist
            assert isinstance(node.condition.stmts[0], dcir.AssignStmt)
            assert isinstance(node.condition.stmts[0].left, dcir.ScalarAccess)
            condition_name = node.condition.stmts[0].left.original_name

            merge_state = self.sdfg.add_state("condition_after")
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(merge_state, edge.dst, edge.data)

            # Evaluate node condition
            init_state = self.sdfg.add_state("condition_init")
            self.sdfg.add_edge(self.state, init_state, dace.InterstateEdge())

            # Promote condition (from init_state) to symbol
            condition_state = self.sdfg.add_state("condition_guard")
            self.sdfg.add_edge(
                init_state,
                condition_state,
                dace.InterstateEdge(assignments=dict(if_condition=condition_name)),
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

        def pop_condition_guard(self) -> None:
            self._pop_last("condition_guard")

        def pop_condition_true(self) -> None:
            self._pop_last("condition_true")

        def pop_condition_false(self) -> None:
            self._pop_last("condition_false")

        def pop_condition_after(self) -> None:
            self._pop_last("condition_after")

        def add_while(self, node: dcir.WhileLoop) -> None:
            """Inserts a while loop after the current state."""
            # Data model validators enforce this to exist
            assert isinstance(node.condition.stmts[0], dcir.AssignStmt)
            assert isinstance(node.condition.stmts[0].left, dcir.ScalarAccess)
            condition_name = node.condition.stmts[0].left.original_name

            after_state = self.sdfg.add_state("while_after")
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(after_state, edge.dst, edge.data)

            # Evaluate loop condition
            init_state = self.sdfg.add_state("while_init")
            self.sdfg.add_edge(self.state, init_state, dace.InterstateEdge())

            # Promote condition (from init_state) to symbol
            guard_state = self.sdfg.add_state("while_guard")
            self.sdfg.add_edge(
                init_state,
                guard_state,
                dace.InterstateEdge(assignments=dict(loop_condition=condition_name)),
            )

            loop_state = self.sdfg.add_state("while_loop")
            self.sdfg.add_edge(
                guard_state, loop_state, dace.InterstateEdge(condition="loop_condition")
            )
            # Loop back to init_state to re-evaluate the loop condition
            self.sdfg.add_edge(loop_state, init_state, dace.InterstateEdge())

            # Exit the loop
            self.sdfg.add_edge(
                guard_state, after_state, dace.InterstateEdge(condition="not loop_condition")
            )

            self.state_stack.append(after_state)
            self.state_stack.append(loop_state)
            self.state_stack.append(guard_state)
            self.state = init_state

        def pop_while_guard(self) -> None:
            self._pop_last("while_guard")

        def pop_while_loop(self) -> None:
            self._pop_last("while_loop")

        def pop_while_after(self) -> None:
            self._pop_last("while_after")

        def _pop_last(self, node_label: str | None = None) -> None:
            if node_label is not None:
                assert self.state_stack[-1].label.startswith(node_label)

            self.state = self.state_stack[-1]
            del self.state_stack[-1]

    def visit_Memlet(self, node: dcir.Memlet, **kwargs: Any) -> None:
        raise RuntimeError("We shouldn't visit memlets anymore. DaCe IR error.")

    def visit_WhileLoop(
        self,
        node: dcir.WhileLoop,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs: Any,
    ) -> None:
        sdfg_ctx.add_while(node)
        assert sdfg_ctx.state.label.startswith("while_init")

        self.visit(node.condition, sdfg_ctx=sdfg_ctx, **kwargs)

        sdfg_ctx.pop_while_guard()
        sdfg_ctx.pop_while_loop()

        for state in node.body:
            self.visit(state, sdfg_ctx=sdfg_ctx, **kwargs)

        sdfg_ctx.pop_while_after()

    def visit_Condition(
        self,
        node: dcir.Condition,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs: Any,
    ) -> None:
        sdfg_ctx.add_condition(node)
        assert sdfg_ctx.state.label.startswith("condition_init")

        self.visit(node.condition, sdfg_ctx=sdfg_ctx, **kwargs)

        sdfg_ctx.pop_condition_guard()
        sdfg_ctx.pop_condition_true()
        for state in node.true_states:
            self.visit(state, sdfg_ctx=sdfg_ctx, **kwargs)

        sdfg_ctx.pop_condition_false()
        for state in node.false_states:
            self.visit(state, sdfg_ctx=sdfg_ctx, **kwargs)

        sdfg_ctx.pop_condition_after()

    def visit_Tasklet(
        self,
        node: dcir.Tasklet,
        *,
        nsdfg_scope: StencilComputationSDFGBuilder.NestedSDFGContext,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **_kwargs: Any,
    ) -> None:
        code = TaskletCodegen.apply_codegen(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            symtable=symtable,
            sdfg=sdfg_ctx.sdfg,
        )

        # We are breaking up vertical loops inside stencils in multiple Tasklets.
        # It might thus happen that we write a "local" scalar in one Tasklet and
        # read it in another Tasklet (downstream).
        # We thus create output connectors for all writes to scalar variables
        # inside Tasklets. And input connectors for all scalar reads unless
        # previously written in the same Tasklet. DaCe's simplify pipeline will get
        # rid of any dead dataflow introduced with this general approach.
        scalar_inputs: dict[str, eve.SymbolRef] = {}
        scalar_outputs: dict[str, eve.SymbolRef] = {}

        # Gather scalar writes in this Tasklet
        for access_node in node.walk_values().if_isinstance(dcir.AssignStmt):
            target_name = access_node.left.name
            if not isinstance(access_node.left, dcir.ScalarAccess) or target_name in scalar_outputs:
                continue

            if access_node.left.original_name is None:
                raise ValueError(
                    "Original name not found for '{access_nodes.left.name}'. DaCeIR error."
                )

            original_name = access_node.left.original_name
            scalar_outputs[target_name] = original_name
            if original_name not in sdfg_ctx.sdfg.arrays:
                sdfg_ctx.sdfg.add_scalar(
                    original_name,
                    dtype=data_type_to_dace_typeclass(access_node.left.dtype),
                    transient=True,
                )

        # Gather scalar reads in this Tasklet
        for access_node in node.walk_values().if_isinstance(dcir.ScalarAccess):
            read_name = access_node.name
            if (
                not access_node.is_target
                and read_name.startswith(prefix.TASKLET_IN)
                and read_name not in scalar_inputs
                and not any(
                    read_name in symbol_map for symbol_map in symtable.maps
                )  # skip defined symbols
            ):
                scalar_inputs[read_name] = access_node.original_name

        tasklet = sdfg_ctx.state.add_tasklet(
            name=node.label,
            code=code,
            inputs=node.input_connectors.union(scalar_inputs.keys()),
            outputs=node.output_connectors.union(scalar_outputs.keys()),
            debuginfo=get_dace_debuginfo(node),
        )

        # Add memlets for scalars access (read/write)
        cache = sdfg_ctx.access_cache.setdefault(sdfg_ctx.state, dict())
        for connector, node_name in scalar_inputs.items():
            access_node = cache.setdefault(node_name, sdfg_ctx.state.add_read(node_name))
            sdfg_ctx.state.add_edge(
                access_node,
                None,
                tasklet,
                connector,
                dace.Memlet.from_array(node_name, sdfg_ctx.sdfg.arrays[node_name]),
            )
        for connector, node_name in scalar_outputs.items():
            cache[node_name] = sdfg_ctx.state.add_write(node_name)
            sdfg_ctx.state.add_edge(
                tasklet,
                connector,
                cache[node_name],
                None,
                dace.Memlet.from_array(node_name, sdfg_ctx.sdfg.arrays[node_name]),
            )

        # Add memlets for field access (read/write)
        for memlet in node.read_memlets:
            # setup access_node (if needed)
            if memlet.field not in cache:
                access_node = sdfg_ctx.state.add_read(memlet.field)
                cache[memlet.field] = access_node

                if memlet.field not in nsdfg_scope.reads:
                    nsdfg_scope.reads.add(memlet.field)

            # connect tasklet (in any case)
            sdfg_ctx.state.add_edge(
                cache[memlet.field],
                None,
                tasklet,
                memlet.connector,
                memlet=_make_dace_memlet(memlet, symtable),
            )

        for memlet in node.write_memlets:
            # setup access_node
            access_node = sdfg_ctx.state.add_access(memlet.field)
            cache[memlet.field] = access_node

            nsdfg_scope.writes.add(memlet.field)

            # connect tasklet
            sdfg_ctx.state.add_edge(
                tasklet,
                memlet.connector,
                access_node,
                None,
                memlet=_make_dace_memlet(memlet, symtable),
            )

    def visit_Range(self, node: dcir.Range, **_kwargs: Any) -> Dict[str, str]:
        start, end = node.interval.to_dace_symbolic()
        return {node.var: str(dace.subsets.Range([(start, end - 1, node.stride)]))}

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        map_scope: StencilComputationSDFGBuilder.DomainMapContext | None = None,
        nsdfg_scope: StencilComputationSDFGBuilder.NestedSDFGContext | None = None,
        **kwargs: Any,
    ) -> None:
        ndranges = {
            k: v
            for index_range in node.index_ranges
            for k, v in self.visit(index_range, **kwargs).items()
        }
        node_name = sdfg_ctx.sdfg.label + "".join(ndranges.keys()) + "_map"
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=node_name,
            ndrange=ndranges,
            schedule=node.schedule.to_dace_schedule(),
            debuginfo=get_dace_debuginfo(node),
        )

        inner_map_scope = StencilComputationSDFGBuilder.DomainMapContext(
            map_entry=map_entry,
            map_exit=map_exit,
            access_cache={},
            last_child=None,
        )

        self.visit(
            node.computations,
            sdfg_ctx=sdfg_ctx,
            map_scope=inner_map_scope,
            symtable=symtable,
            **kwargs,
        )

        if map_scope is None:
            # sanity check
            if nsdfg_scope is None:
                raise ValueError(
                    "Expected top-level nested SDFG scope around non-nested DomainMap."
                )

            # There's only ever one outer-most DomainMap because we have one nested SDFG
            # per vertical domain. We thus don't need to care about read after write and
            # other complications at this level.
            for connector in map_entry.in_connectors:
                field_name = connector.removeprefix(prefix.PASSTHROUGH_IN)

                # Get the dcir.Memlet
                candidates = [
                    m for m in node.read_memlets + node.write_memlets if m.field == field_name
                ]
                if not len(candidates) == 1:
                    raise ValueError(f"Pre-computed dcir.Memlet not found for {field_name}.")

                access_node = sdfg_ctx.state.add_read(field_name)
                sdfg_ctx.state.add_edge(
                    access_node,
                    None,
                    map_entry,
                    connector,
                    _make_dace_memlet(candidates[0], symtable),
                )

                nsdfg_scope.reads.add(field_name)

            for connector in map_exit.out_connectors:
                field_name = connector.removeprefix(prefix.PASSTHROUGH_OUT)

                # Get the dcir.Memlet
                candidates = [m for m in node.write_memlets if m.field == field_name]
                if not len(candidates) == 1:
                    raise ValueError(f"Pre-computed dcir.Memlet not found for {field_name}.")

                access_node = sdfg_ctx.state.add_write(field_name)
                sdfg_ctx.state.add_edge(
                    map_exit,
                    connector,
                    access_node,
                    None,
                    _make_dace_memlet(candidates[0], symtable),
                )

                nsdfg_scope.writes.add(field_name)

            if sdfg_ctx.state.in_degree(map_exit) < 1:
                # if map_exit isn't connected yet, connect with an empty memlet to the last child
                sdfg_ctx.state.add_nedge(inner_map_scope.last_child, map_exit, dace.Memlet())

            return

        # in case we are inside another DomainMap
        map_scope.last_child = map_exit

        # Handle reads
        for connector in map_entry.in_connectors:
            field_name = connector.removeprefix(prefix.PASSTHROUGH_IN)

            # Get the dcir.Memlet
            candidates = [
                m for m in node.read_memlets + node.write_memlets if m.field == field_name
            ]
            if not len(candidates) == 1:
                raise ValueError(f"Pre-computed dcir.Memlet not found for {field_name}.")

            if field_name in map_scope.access_cache:
                # cached read
                cached_access = map_scope.access_cache[field_name]
                sdfg_ctx.state.add_edge(
                    cached_access[0],
                    cached_access[1],
                    map_entry,
                    connector,
                    _make_dace_memlet(candidates[0], symtable),
                )

                continue

            # new read
            out_connector = f"{prefix.PASSTHROUGH_OUT}{field_name}"
            new_in = map_scope.map_entry.add_in_connector(connector)
            new_out = map_scope.map_entry.add_out_connector(out_connector)
            assert new_in
            assert new_out

            sdfg_ctx.state.add_edge(
                map_scope.map_entry,
                out_connector,
                map_entry,
                connector,
                _make_dace_memlet(candidates[0], symtable),
            )

            map_scope.access_cache[field_name] = (map_scope.map_entry, out_connector)

        # If the first child happens to make no input connectors, add an empty memlet instead to preserve order.
        if sdfg_ctx.state.out_degree(map_scope.map_entry) < 1:
            sdfg_ctx.state.add_nedge(map_scope.map_entry, map_entry, dace.Memlet())

        # to be removed again, just a sanity check
        assert sdfg_ctx.state.out_degree(map_scope.map_entry) >= 1

        # Handle writes
        for connector in map_exit.out_connectors:
            field_name = connector.removeprefix(prefix.PASSTHROUGH_OUT)

            # Get the dcir.Memlet
            candidates = [m for m in node.write_memlets if m.field == field_name]
            if not len(candidates) == 1:
                raise ValueError(f"Pre-computed dcir.Memlet not found for {field_name}.")

            # always create a new access node
            access_node = sdfg_ctx.state.add_write(field_name)
            sdfg_ctx.state.add_edge(
                map_exit, connector, access_node, None, _make_dace_memlet(candidates[0], symtable)
            )
            in_connector = f"{prefix.PASSTHROUGH_IN}{field_name}"

            if field_name in map_scope.access_cache:
                # write after write -> add an empty memlet to ensure order of operations
                sdfg_ctx.state.add_nedge(
                    map_scope.access_cache[field_name][0], map_entry, dace.Memlet()
                )
            else:
                # new write
                new_in = map_scope.map_exit.add_in_connector(in_connector)
                new_out = map_scope.map_exit.add_out_connector(connector)
                assert new_in
                assert new_out

                map_scope.access_cache[field_name] = (access_node, None)

            sdfg_ctx.state.add_edge(
                access_node,
                None,
                map_scope.map_exit,
                in_connector,
                _make_dace_memlet(candidates[0], symtable),
            )

        if sdfg_ctx.state.in_degree(map_exit) < 1:
            # if map_exit isn't connected yet, connect with an empty memlet to the last child
            sdfg_ctx.state.add_nedge(inner_map_scope.last_child, map_exit, dace.Memlet())

        # Just for testing - to be removed
        assert node

    def visit_DomainLoop(
        self,
        node: dcir.DomainLoop,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs: Any,
    ) -> None:
        sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs: Any,
    ) -> None:
        sdfg_ctx.add_state()

        for computation in node.computations:
            self.visit(computation, sdfg_ctx=sdfg_ctx, **kwargs)

    def visit_FieldDecl(
        self,
        node: dcir.FieldDecl,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        non_transients: Set[eve.SymbolRef],
        **kwargs: Any,
    ) -> None:
        assert len(node.strides) == len(node.shape)
        sdfg_ctx.sdfg.add_array(
            node.name,
            shape=node.shape,
            strides=[dace.symbolic.pystr_to_symbolic(s) for s in node.strides],
            dtype=data_type_to_dace_typeclass(node.dtype),
            storage=node.storage.to_dace_storage(),
            transient=node.name not in non_transients,
            debuginfo=get_dace_debuginfo(node),
        )

    def visit_SymbolDecl(
        self,
        node: dcir.SymbolDecl,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs: Any,
    ) -> None:
        if node.name not in sdfg_ctx.sdfg.symbols:
            sdfg_ctx.sdfg.add_symbol(node.name, stype=data_type_to_dace_typeclass(node.dtype))

    def visit_NestedSDFG(
        self,
        node: dcir.NestedSDFG,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext | None = None,
        map_scope: StencilComputationSDFGBuilder.DomainMapContext | None = None,
        # not actually used. Just pulling it out of kwargs here for easy replacement with inn_nsdfg_scope inside the function
        nsdfg_scope: StencilComputationSDFGBuilder.NestedSDFGContext | None = None,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **kwargs: Any,
    ) -> dace.nodes.NestedSDFG:
        # setup inner SDFG and context
        sdfg = dace.SDFG(node.label)
        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg, state=sdfg.add_state(is_start_block=True)
        )
        inner_ndsfg_scope = StencilComputationSDFGBuilder.NestedSDFGContext(
            reads=set(), writes=set()
        )

        self.visit(
            node.field_decls,
            sdfg_ctx=inner_sdfg_ctx,
            nsdfg_scope=inner_ndsfg_scope,
            # Transients inside a NestedSDFG are non-transients.
            non_transients={memlet.connector for memlet in node.read_memlets + node.write_memlets},
            symtable=symtable,
            **kwargs,
        )
        self.visit(
            node.symbol_decls,
            sdfg_ctx=inner_sdfg_ctx,
            nsdfg_scope=inner_ndsfg_scope,
            symtable=symtable,
            **kwargs,
        )
        symbol_mapping = {decl.name: decl.to_dace_symbol() for decl in node.symbol_decls}

        # visit child nodes
        for computation_state in node.states:
            self.visit(
                computation_state,
                sdfg_ctx=inner_sdfg_ctx,
                nsdfg_scope=inner_ndsfg_scope,
                symtable=symtable,
                **kwargs,
            )

        if sdfg_ctx is None:
            # This is the top level nested SDFG, i.e. the one used to replace the library node in the expansion.
            if nsdfg_scope is not None or map_scope is not None:
                raise ValueError(
                    "Top-level nested SDFG shouldn't be contained in a scope. DaCeIR error."
                )

            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs=inner_ndsfg_scope.reads,
                outputs=inner_ndsfg_scope.writes,
                symbol_mapping=symbol_mapping,
            )
            return nsdfg

        # We are inside another nested SDFG:
        if map_scope is None:
            raise ValueError("Expected map_scope around nested SDFG.")

        # Add nested SDFG
        nsdfg = sdfg_ctx.state.add_nested_sdfg(
            sdfg=sdfg,
            parent=sdfg_ctx.sdfg,
            inputs=inner_ndsfg_scope.reads,
            outputs=inner_ndsfg_scope.writes,
            symbol_mapping=symbol_mapping,
        )

        # Book-keep last_child
        map_scope.last_child = nsdfg

        # Handle reads
        for field_name in inner_ndsfg_scope.reads:
            # Get the dcir.Memlet
            candidates = [
                m for m in node.read_memlets + node.write_memlets if m.field == field_name
            ]
            if not len(candidates) == 1:
                raise ValueError(f"Pre-computed dcir.Memlet not found for {field_name}.")

            # cached read
            if field_name in map_scope.access_cache:
                src_node, src_connector = map_scope.access_cache[field_name]
                sdfg_ctx.state.add_edge(
                    src_node,
                    src_connector,
                    nsdfg,
                    field_name,
                    _make_dace_memlet(candidates[0], symtable),
                )
                continue

            # new read
            out_connector = f"{prefix.PASSTHROUGH_OUT}{field_name}"
            new_in = map_scope.map_entry.add_in_connector(f"{prefix.PASSTHROUGH_IN}{field_name}")
            new_out = map_scope.map_entry.add_out_connector(out_connector)
            assert new_in
            assert new_out
            sdfg_ctx.state.add_edge(
                map_scope.map_entry,
                out_connector,
                nsdfg,
                field_name,
                _make_dace_memlet(candidates[0], symtable),
            )
            map_scope.access_cache[field_name] = (map_scope.map_entry, out_connector)

        # If the first child happens to make no input connectors, add an empty memlet instead to preserve order.
        if sdfg_ctx.state.out_degree(map_scope.map_entry) < 1:
            sdfg_ctx.state.add_nedge(map_scope.map_entry, nsdfg, dace.Memlet())

        # to be removed again, just a sanity check
        assert sdfg_ctx.state.out_degree(map_scope.map_entry) >= 1

        # Handle writes
        for field_name in inner_ndsfg_scope.writes:
            # Get the dcir.Memlet
            candidates = [m for m in node.write_memlets if m.field == field_name]
            if not len(candidates) == 1:
                raise ValueError(f"Pre-computed dcir.Memlet not found for {field_name}.")

            # always create new access node
            access_node = nodes.AccessNode(field_name)
            sdfg_ctx.state.add_edge(
                nsdfg, field_name, access_node, None, _make_dace_memlet(candidates[0], symtable)
            )

            # optionally add in/out connectors on map_scope
            in_connector = f"{prefix.PASSTHROUGH_IN}{field_name}"

            if field_name in map_scope.access_cache:
                # write after write -> add an empty memlet to ensure order of operations
                sdfg_ctx.state.add_nedge(
                    map_scope.access_cache[field_name][0], nsdfg, dace.Memlet()
                )
            else:
                # new write
                new_in = map_scope.map_exit.add_in_connector(in_connector)
                new_out = map_scope.map_exit.add_out_connector(
                    f"{prefix.PASSTHROUGH_OUT}{field_name}"
                )
                assert new_in
                assert new_out
                map_scope.access_cache[field_name] = (access_node, None)

            sdfg_ctx.state.add_edge(
                access_node,
                None,
                map_scope.map_exit,
                in_connector,
                _make_dace_memlet(candidates[0], symtable),
            )

        return nsdfg
