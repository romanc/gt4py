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
from gt4py.cartesian.gtc.dace.utils import get_dace_debuginfo, make_dace_subset


def _node_name_from_connector(connector: str) -> str:
    if not connector.startswith(prefix.TASKLET_IN) and not connector.startswith(prefix.TASKLET_OUT):
        raise ValueError(
            f"Connector {connector} doesn't follow the in ({prefix.TASKLET_IN}) or out ({prefix.TASKLET_OUT}) prefix rule"
        )
    return connector.removeprefix(prefix.TASKLET_OUT).removeprefix(prefix.TASKLET_IN)


class StencilComputationSDFGBuilder(eve.VisitorWithSymbolTableTrait):
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
        df_scope: dict[str, dict[eve.SymbolRef, nodes.AccessNode]],
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
            sdfg_ctx.state.add_memlet_path(
                access_node, tasklet, dst_conn=connector, memlet=dace.Memlet(data=node_name)
            )
        for connector, node_name in scalar_outputs.items():
            access_node = cache.setdefault(node_name, sdfg_ctx.state.add_write(node_name))
            sdfg_ctx.state.add_memlet_path(
                tasklet, access_node, src_conn=connector, memlet=dace.Memlet(data=node_name)
            )

        # Add memlets for field access (read/write)
        for memlet in node.read_memlets:
            # setup access_node if needed
            if memlet.field not in cache:
                access_node = sdfg_ctx.state.add_read(memlet.field)
                cache[memlet.field] = access_node

                if memlet.field not in df_scope["input"]:
                    df_scope["input"][memlet.field] = access_node

            # connect tasklet (in any case)
            field_decl = symtable[memlet.field]
            assert isinstance(field_decl, dcir.FieldDecl)
            sdfg_ctx.state.add_memlet_path(
                cache[memlet.field],
                tasklet,
                dst_conn=memlet.connector,
                memlet=dace.Memlet(
                    memlet.field,
                    subset=make_dace_subset(
                        field_decl.access_info, memlet.access_info, field_decl.data_dims
                    ),
                    dynamic=field_decl.is_dynamic,
                ),
            )

        for memlet in node.write_memlets:
            # setup access_node
            access_node = sdfg_ctx.state.add_access(memlet.field)
            cache[memlet.field] = access_node

            df_scope["output"][memlet.field] = access_node

            # connect tasklet
            field_decl = symtable[memlet.field]
            assert isinstance(field_decl, dcir.FieldDecl)
            sdfg_ctx.state.add_memlet_path(
                tasklet,
                access_node,
                src_conn=memlet.connector,
                memlet=dace.Memlet(
                    memlet.field,
                    subset=make_dace_subset(
                        field_decl.access_info, memlet.access_info, field_decl.data_dims
                    ),
                    dynamic=field_decl.is_dynamic,
                ),
            )

    def visit_Range(self, node: dcir.Range, **kwargs: Any) -> Dict[str, str]:
        start, end = node.interval.to_dace_symbolic()
        return {node.var: str(dace.subsets.Range([(start, end - 1, node.stride)]))}

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        df_scope: dict[str, dict[eve.SymbolRef, nodes.AccessNode]] | None,
        **kwargs: Any,
    ) -> None:
        ndranges = {
            k: v
            for index_range in node.index_ranges
            for k, v in self.visit(index_range, **kwargs).items()
        }
        name = sdfg_ctx.sdfg.label + "".join(ndranges.keys()) + "_map"
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=name,
            ndrange=ndranges,
            schedule=node.schedule.to_dace_schedule(),
            debuginfo=get_dace_debuginfo(node),
        )

        inner_df_scope: dict[str, dict[eve.SymbolRef, nodes.AccessNode]] = {
            "input": dict(),
            "output": dict(),
        }

        for scope_node in node.computations:
            self.visit(scope_node, sdfg_ctx=sdfg_ctx, df_scope=inner_df_scope, **kwargs)

        # add in/out connectors to map_entry / map_exit nodes from inner_df_scope

        # add empty edges based on inner_df_scope

        # - if df_scope is not None:
        #     "relay" inner_df_scope to df_scope from outside (adding access nodes in the process)

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
        df_scope: dict[str, dict[eve.SymbolRef, nodes.AccessNode]] | None,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs: Any,
    ) -> dace.nodes.NestedSDFG:
        sdfg = dace.SDFG(node.label)

        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg, state=sdfg.add_state(is_start_block=True)
        )
        inner_df_scope: dict[str, dict[eve.SymbolRef, nodes.AccessNode]] = {
            "input": dict(),
            "output": dict(),
        }

        self.visit(
            node.field_decls,
            sdfg_ctx=inner_sdfg_ctx,
            df_scope=inner_df_scope,
            non_transients={memlet.connector for memlet in node.read_memlets + node.write_memlets},
            **kwargs,
        )
        self.visit(node.symbol_decls, sdfg_ctx=inner_sdfg_ctx, df_scope=inner_df_scope, **kwargs)
        symbol_mapping = {decl.name: decl.to_dace_symbol() for decl in node.symbol_decls}

        for computation_state in node.states:
            self.visit(
                computation_state,
                sdfg_ctx=inner_sdfg_ctx,
                df_scope=inner_df_scope,
                symtable=symtable,
                **kwargs,
            )

        # add in/out connectors to nested SDFG based on inner_df_scope
        # -> then think about moving the whole thing to the DaCe IR level maybe?
        #    or why would we have {input,output}_connectors on the node then?

        # add empty edges based on inner_df_scope

        # - if df_scope is not None:
        #     "relay" inner_df_scope to df_scope from outside (adding access_nodes in the process)

        if sdfg_ctx is not None and df_scope is not None:
            # we are inside a nested SDFG
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=node.input_connectors,
                outputs=node.output_connectors,
                symbol_mapping=symbol_mapping,
            )
            return nsdfg

        return dace.nodes.NestedSDFG(
            label=sdfg.label,
            sdfg=sdfg,
            inputs={memlet.connector for memlet in node.read_memlets},
            outputs={memlet.connector for memlet in node.write_memlets},
            symbol_mapping=symbol_mapping,
        )
