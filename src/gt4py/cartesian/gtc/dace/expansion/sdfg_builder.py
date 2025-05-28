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


def _node_name_from_connector(connector: str) -> str:
    if not connector.startswith(prefix.TASKLET_IN) and not connector.startswith(prefix.TASKLET_OUT):
        raise ValueError(
            f"Connector {connector} doesn't follow the in ({prefix.TASKLET_IN}) or out ({prefix.TASKLET_OUT}) prefix rule"
        )
    return connector.removeprefix(prefix.TASKLET_OUT).removeprefix(prefix.TASKLET_IN)


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
        # per DomainMap access cache
        cache: dict[str, dace.nodes.AccessNode]
        # input: (AccessNode) -> [(ScopeNode, connector, Memlet)] because Memlet is not hashable
        # output: (ScopeNode, connector) -> (AccessNode, Memlet)
        connect: dict[
            tuple[dace.nodes.AccessNode] | tuple[dace.nodes.Node, str],
            list[tuple[dace.nodes.Node, str, dace.Memlet]]
            | tuple[dace.nodes.AccessNode, dace.Memlet],
        ]

        first_child: dace.nodes.Node | None
        last_child: dace.nodes.Node | None

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
        **kwargs: Any,
    ) -> None:
        code = TaskletCodegen.apply_codegen(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            symtable=symtable,
            sdfg=sdfg_ctx.sdfg,
        )

        # We are breaking up vertical loops inside stencils in multiple Tasklets
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
            # setup access_node if needed
            if memlet.field not in cache:
                access_node = sdfg_ctx.state.add_read(memlet.field)
                cache[memlet.field] = access_node

                if memlet.field not in nsdfg_scope.reads:
                    nsdfg_scope.reads.add(memlet.field)

            # connect tasklet (in any case)
            sdfg_ctx.state.add_memlet_path(
                cache[memlet.field],
                tasklet,
                dst_conn=memlet.connector,
                memlet=_make_dace_memlet(memlet, symtable),
            )

        for memlet in node.write_memlets:
            # setup access_node
            access_node = sdfg_ctx.state.add_access(memlet.field)
            cache[memlet.field] = access_node

            nsdfg_scope.writes.add(memlet.field)

            # connect tasklet
            sdfg_ctx.state.add_memlet_path(
                tasklet,
                access_node,
                src_conn=memlet.connector,
                memlet=_make_dace_memlet(memlet, symtable),
            )

    def visit_Range(self, node: dcir.Range, **kwargs: Any) -> Dict[str, str]:
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
            cache={},
            connect={},
            first_child=None,
            last_child=None,
        )

        self.visit(
            node.computations,
            sdfg_ctx=sdfg_ctx,
            map_scope=inner_map_scope,
            symtable=symtable,
            **kwargs,
        )

        if map_scope is not None:
            if map_scope.first_child is None:
                map_scope.first_child = map_entry
            map_scope.last_child = map_exit

        cache = sdfg_ctx.access_cache.setdefault(sdfg_ctx.state, {})

        previously_written: dict[str, dace.nodes.AccessNode] = {}
        for _index, (src, dst) in enumerate(inner_map_scope.connect.items()):
            # read access
            if isinstance(src[0], dace.nodes.AccessNode):
                assert isinstance(dst, list)
                field_name = src[0].data
                in_connector = f"{prefix.PASSTHROUGH_IN}{field_name}"
                out_connector = f"{prefix.PASSTHROUGH_OUT}{field_name}"
                map_entry.add_in_connector(in_connector)
                map_entry.add_out_connector(out_connector)

                # connect "inwards"
                for inner_node, inner_connector, inner_memlet in dst:
                    if field_name in previously_written:
                        # read from access node
                        sdfg_ctx.state.add_edge(
                            previously_written[field_name],
                            None,
                            inner_node,
                            inner_connector,
                            _make_dace_memlet(inner_memlet, symtable),
                            # dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]), # noqa
                        )
                    else:
                        # connect directly to map_entry node
                        sdfg_ctx.state.add_edge(
                            map_entry,
                            out_connector,
                            inner_node,
                            inner_connector,
                            _make_dace_memlet(inner_memlet, symtable),
                            # dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]), # noqa
                        )

                # connect "outwards"
                if map_scope is not None:
                    # add new access node (if not cached)
                    access_node = map_scope.cache.setdefault(
                        field_name, dace.nodes.AccessNode(field_name)
                    )

                    # connect (cached) access -> (map_entry, in_connector, memlet)
                    candidates = [m for m in node.read_memlets if m.field == field_name]
                    assert len(candidates) == 1
                    if (access_node,) in map_scope.connect:
                        map_scope.connect[(access_node,)].append(  # type: ignore
                            (map_entry, in_connector, candidates[0])
                        )
                    else:
                        map_scope.connect[(access_node,)] = [
                            (map_entry, in_connector, candidates[0])
                        ]
                    continue

                access_node = cache.setdefault(field_name, sdfg_ctx.state.add_read(field_name))
                sdfg_ctx.state.add_edge(
                    access_node,
                    None,
                    map_entry,
                    in_connector,
                    dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]),
                )
                if nsdfg_scope is None:
                    raise ValueError("Expected (top-level) nested SDFG scope. DaCe IR error.")
                nsdfg_scope.reads.add(field_name)
                continue

            # write access
            if not isinstance(dst[0], dace.nodes.AccessNode):
                raise ValueError(
                    f"Expected (write) access node, got {type(dst[0])} instead. DaCe IR error."
                )
            assert len(src) == 2
            assert isinstance(dst, tuple)

            field_name = dst[0].data
            in_connector = f"{prefix.PASSTHROUGH_IN}{field_name}"
            out_connector = f"{prefix.PASSTHROUGH_OUT}{field_name}"
            map_exit.add_in_connector(in_connector)
            map_exit.add_out_connector(out_connector)

            # connect "inwards"
            handled = False
            # -/a for j_index, item in enumerate(inner_map_scope.connect.items()):
            # -/a     if j_index <= index or handled:
            # -/a         continue
            # -/a     if item[1][0] == dst[0]:
            # -/a         # write after write
            # -/a         raise NotImplementedError("Write after write in df_scope. Freak out")
            # -/a     if item[0][0] == dst[0]:
            # -/a         # read after write, add an access node
            # -/a         node = cache.setdefault(field_name, sdfg_ctx.state.add_access(field_name))
            # -/a         previously_written[field_name] = node
            # -/a         sdfg_ctx.state.add_edge(
            # -/a             src[0],
            # -/a             src[1],
            # -/a             node,
            # -/a             None,
            # -/a             _make_dace_memlet(dst[1], symtable),
            # -/a             # dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]),
            # -/a         )
            # -/a         sdfg_ctx.state.add_edge(
            # -/a             node,
            # -/a             None,
            # -/a             map_exit,
            # -/a             in_connector,
            # -/a             _make_dace_memlet(dace.Memlet.from_memlet(dst[1]), symtable),
            # -/a             # dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]),
            # -/a         )
            # -/a         handled = True

            if not handled:
                sdfg_ctx.state.add_edge(
                    src[0],
                    src[1],
                    map_exit,
                    in_connector,
                    _make_dace_memlet(dst[1], symtable),
                    # dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]), # noqa
                )

            # connect "outwards"
            if map_scope is not None:
                # add new access_node
                map_scope.cache[field_name] = dace.nodes.AccessNode(field_name)

                # connect (map_exit, out_connector) -> new access_node
                candidates = [m for m in node.write_memlets if m.field == field_name]
                assert len(candidates) == 1
                map_scope.connect[(map_exit, out_connector)] = (
                    map_scope.cache[field_name],
                    candidates[0],
                )
                continue

            access_node = sdfg_ctx.state.add_write(field_name)
            cache[field_name] = access_node
            sdfg_ctx.state.add_edge(
                map_exit,
                out_connector,
                access_node,
                None,
                dace.Memlet.from_array(field_name, sdfg_ctx.sdfg.arrays[field_name]),
            )

            if nsdfg_scope is None:
                raise ValueError("Expected (top-level) nested SDFG scope. DaCe IR error.")
            nsdfg_scope.writes.add(field_name)

        if not map_entry.out_connectors:
            assert inner_map_scope.first_child is not None, "Expected first_child to be defined."
            sdfg_ctx.state.add_nedge(map_entry, inner_map_scope.first_child, dace.Memlet())
        if not map_exit.in_connectors:
            assert inner_map_scope.last_child is not None, "Expected last_child to be defined."
            sdfg_ctx.state.add_nedge(inner_map_scope.last_child, map_exit, dace.Memlet())
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
        nsdfg_scope: StencilComputationSDFGBuilder.NestedSDFGContext | None = None,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs: Any,
    ) -> dace.nodes.NestedSDFG:
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
            non_transients={memlet.connector for memlet in node.read_memlets + node.write_memlets},
            **kwargs,
        )
        self.visit(
            node.symbol_decls, sdfg_ctx=inner_sdfg_ctx, nsdfg_scope=inner_ndsfg_scope, **kwargs
        )
        symbol_mapping = {decl.name: decl.to_dace_symbol() for decl in node.symbol_decls}

        for computation_state in node.states:
            self.visit(
                computation_state,
                sdfg_ctx=inner_sdfg_ctx,
                nsdfg_scope=inner_ndsfg_scope,
                symtable=symtable,
                **kwargs,
            )

        if sdfg_ctx is not None:
            if map_scope is None:
                raise ValueError("Expected map_scope around nested SDFG.")

            # we are inside a nested SDFG
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=inner_ndsfg_scope.reads,
                outputs=inner_ndsfg_scope.writes,
                symbol_mapping=symbol_mapping,
            )

            if map_scope.first_child is None:
                map_scope.first_child = nsdfg
            map_scope.last_child = nsdfg

            for node_name in inner_ndsfg_scope.reads:
                # add new access node (if not cached)
                access_node = map_scope.cache.setdefault(
                    node_name, dace.nodes.AccessNode(node_name)
                )

                # connect (cached) access -> (nsdfg, node_name)
                candidates = [m for m in node.read_memlets if m.field == node_name]
                assert len(candidates) == 1
                if (access_node,) in map_scope.connect:
                    map_scope.connect[(access_node,)].append(  # type: ignore
                        (nsdfg, node_name, candidates[0])
                    )
                else:
                    map_scope.connect[(access_node,)] = [(nsdfg, node_name, candidates[0])]

            for node_name in inner_ndsfg_scope.writes:
                # always create new access node
                map_scope.cache[node_name] = dace.nodes.AccessNode(node_name)

                # connect (nsdfg, node_name) -> new access_node
                candidates = [m for m in node.write_memlets if m.field == node_name]
                assert len(candidates) == 1
                map_scope.connect[(nsdfg, node_name)] = (map_scope.cache[node_name], candidates[0])

            return nsdfg

        # top level nested SDFG (the one used to replace the library node in the expansion)
        nsdfg = dace.nodes.NestedSDFG(
            label=sdfg.label,
            sdfg=sdfg,
            inputs=inner_ndsfg_scope.reads,
            outputs=inner_ndsfg_scope.writes,
            symbol_mapping=symbol_mapping,
        )
        return nsdfg
