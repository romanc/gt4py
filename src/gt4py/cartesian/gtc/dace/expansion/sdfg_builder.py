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
from typing import Any, ChainMap, Dict, List, Optional, Set, Tuple, Union

import dace
import dace.data
import dace.library
import dace.subsets

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace.expansion.tasklet_codegen import TaskletCodegen
from gt4py.cartesian.gtc.dace.expansion.utils import get_dace_debuginfo
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gt4py.cartesian.gtc.dace.utils import make_dace_subset


class StencilComputationSDFGBuilder(eve.VisitorWithSymbolTableTrait):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        state_stack: List[dace.SDFGState] = dataclasses.field(default_factory=list)

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

        def add_condition(self, tmp_condition_name: str):
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
                dace.InterstateEdge(assignments=dict(if_condition=tmp_condition_name)),
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
            self.state = init_state
            return self

        def pop_condition_true(self):
            self._pop_last("condition_true")

        def pop_condition_false(self):
            self._pop_last("condition_false")

        def pop_condition_after(self):
            self._pop_last("condition_after")

        def add_while(self, tmp_condition_name: str):
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
                dace.InterstateEdge(assignments=dict(loop_condition=tmp_condition_name)),
            )

            loop_state = self.sdfg.add_state("while_loop")
            self.sdfg.add_edge(
                guard_state, loop_state, dace.InterstateEdge(condition="loop_condition")
            )
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

    def visit_Memlet(
        self,
        node: dcir.Memlet,
        *,
        scope_node: dcir.ComputationNode,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        connector_prefix="",
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
    ) -> None:
        field_decl = symtable[node.field]
        assert isinstance(field_decl, dcir.FieldDecl)
        memlet = dace.Memlet(
            node.field,
            subset=make_dace_subset(field_decl.access_info, node.access_info, field_decl.data_dims),
            dynamic=field_decl.is_dynamic,
        )
        if node.is_read:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[memlet.data],
                scope_node,
                connector_prefix + node.connector,
                memlet,
            )
        if node.is_write:
            sdfg_ctx.state.add_edge(
                scope_node,
                connector_prefix + node.connector,
                *node_ctx.output_node_and_conns[memlet.data],
                memlet,
            )

    @classmethod
    def _add_empty_edges(
        cls,
        entry_node: dace.nodes.Node,
        exit_node: dace.nodes.Node,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
    ) -> None:
        if not sdfg_ctx.state.in_degree(entry_node) and None in node_ctx.input_node_and_conns:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[None], entry_node, None, dace.Memlet()
            )
        if not sdfg_ctx.state.out_degree(exit_node) and None in node_ctx.output_node_and_conns:
            sdfg_ctx.state.add_edge(
                exit_node, None, *node_ctx.output_node_and_conns[None], dace.Memlet()
            )

    def visit_WhileLoop(
        self,
        node: dcir.WhileLoop,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **kwargs: Any,
    ) -> None:
        tmp_condition_name = f"tmp_conditional_{id(node)}"
        sdfg_ctx.add_while(tmp_condition_name)
        assert sdfg_ctx.state.label.startswith("while_init")
        self._add_condition_evaluation_tasklet(
            tmp_condition_name,
            node,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            symtable=symtable,
            **kwargs,
        )

        # Cleanup: we might not need this state on the stack anymore
        sdfg_ctx.pop_while_guard()
        assert sdfg_ctx.state.label.startswith("while_guard")

        sdfg_ctx.pop_while_loop()
        assert sdfg_ctx.state.label.startswith("while_loop")
        for state in node.body:
            self.visit(state, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, symtable=symtable, **kwargs)

        sdfg_ctx.pop_while_after()
        assert sdfg_ctx.state.label.startswith("while_after")

    def visit_Condition(
        self,
        node: dcir.Condition,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **kwargs: Any,
    ) -> None:
        tmp_condition_name = f"tmp_conditional_{id(node)}"
        sdfg_ctx.add_condition(tmp_condition_name)
        assert sdfg_ctx.state.label.startswith("condition_init")
        self._add_condition_evaluation_tasklet(
            tmp_condition_name,
            node,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            symtable=symtable,
            **kwargs,
        )

        sdfg_ctx.pop_condition_true()
        assert sdfg_ctx.state.label.startswith("condition_true")
        for state in node.true_state:
            self.visit(state, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, symtable=symtable, **kwargs)

        sdfg_ctx.pop_condition_false()
        for state in node.false_state:
            self.visit(state, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, symtable=symtable, **kwargs)

        sdfg_ctx.pop_condition_after()
        assert sdfg_ctx.state.label.startswith("condition_after")

    def visit_Tasklet(
        self,
        node: dcir.Tasklet,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **kwargs,
    ) -> None:
        code = TaskletCodegen.apply_codegen(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            sdfg_ctx=sdfg_ctx,
            symtable=symtable,
        )

        tasklet_inputs: Set[str] = set()
        tasklet_outputs: Set[str] = set()

        # merge write_memlets with writes of local scalar declarations (as defined by node.decls)
        for access_node in node.walk_values().if_isinstance(dcir.AssignStmt):
            target_name = access_node.left.name
            matches = [declaration for declaration in node.decls if declaration.name == target_name]

            if len(matches) > 1:
                raise RuntimeError(
                    "Found more than one matching declaration for '%s'" % target_name
                )

            if len(matches) > 0:
                tasklet_outputs.add(target_name)
                sdfg_ctx.sdfg.add_scalar(
                    target_name, dtype=data_type_to_dace_typeclass(matches[0].dtype), transient=True
                )

        # merge read_memlets with reads of local scalars (unless written in the same tasklet)
        for access_node in node.walk_values().if_isinstance(dcir.ScalarAccess):
            read_name = access_node.name
            locally_declared = (
                len([declaration for declaration in node.decls if declaration.name == read_name])
                > 0
            )
            field_access = (
                len(
                    set(
                        memlet.connector
                        for memlet in [*node.read_memlets, *node.write_memlets]
                        if memlet.connector == read_name
                    )
                )
                > 0
            )

            if not locally_declared and not field_access:
                tasklet_inputs.add(read_name)

        tasklet = sdfg_ctx.state.add_tasklet(
            name=f"{sdfg_ctx.sdfg.label}_Tasklet",
            code=code,
            inputs=tasklet_inputs.union(set(memlet.connector for memlet in node.read_memlets)),
            outputs=tasklet_outputs.union(set(memlet.connector for memlet in node.write_memlets)),
            debuginfo=get_dace_debuginfo(node),
        )

        # add memlets for local scalars into / out of tasklet
        for output_name in tasklet_outputs:
            access_node = sdfg_ctx.state.add_access(output_name)
            write_memlet = dace.Memlet(output_name)
            sdfg_ctx.state.add_memlet_path(
                tasklet, access_node, src_conn=output_name, memlet=write_memlet
            )
        for input_name in tasklet_inputs:
            access_node = sdfg_ctx.state.add_access(input_name)
            read_memlet = dace.Memlet(input_name)
            sdfg_ctx.state.add_memlet_path(
                access_node, tasklet, dst_conn=input_name, memlet=read_memlet
            )

        self.visit(
            node.read_memlets,
            scope_node=tasklet,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            symtable=symtable,
            **kwargs,
        )
        self.visit(
            node.write_memlets,
            scope_node=tasklet,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            symtable=symtable,
            **kwargs,
        )
        StencilComputationSDFGBuilder._add_empty_edges(
            tasklet, tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
        )

    def visit_Range(self, node: dcir.Range, **kwargs) -> Dict[str, str]:
        start, end = node.interval.to_dace_symbolic()
        return {node.var: str(dace.subsets.Range([(start, end - 1, node.stride)]))}

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs,
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

        for scope_node in node.computations:
            input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
            output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
            for field in set(memlet.field for memlet in scope_node.read_memlets):
                map_entry.add_in_connector("IN_" + field)
                map_entry.add_out_connector("OUT_" + field)
                input_node_and_conns[field] = (map_entry, "OUT_" + field)
            for field in set(memlet.field for memlet in scope_node.write_memlets):
                map_exit.add_in_connector("IN_" + field)
                map_exit.add_out_connector("OUT_" + field)
                output_node_and_conns[field] = (map_exit, "IN_" + field)
            if not input_node_and_conns:
                input_node_and_conns[None] = (map_entry, None)
            if not output_node_and_conns:
                output_node_and_conns[None] = (map_exit, None)
            inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=input_node_and_conns,
                output_node_and_conns=output_node_and_conns,
            )
            self.visit(scope_node, sdfg_ctx=sdfg_ctx, node_ctx=inner_node_ctx, **kwargs)

        self.visit(
            node.read_memlets,
            scope_node=map_entry,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="IN_",
            **kwargs,
        )
        self.visit(
            node.write_memlets,
            scope_node=map_exit,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="OUT_",
            **kwargs,
        )
        StencilComputationSDFGBuilder._add_empty_edges(
            map_entry, map_exit, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
        )

    def visit_DomainLoop(
        self,
        node: dcir.DomainLoop,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs,
    ) -> None:
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        **kwargs,
    ) -> None:
        for computation in node.computations:
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, **kwargs)

    def visit_FieldDecl(
        self,
        node: dcir.FieldDecl,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        non_transients: Set[eve.SymbolRef],
        **kwargs,
    ) -> None:
        assert len(node.strides) == len(node.shape)
        sdfg_ctx.sdfg.add_array(
            node.name,
            shape=node.shape,
            strides=[dace.symbolic.pystr_to_symbolic(s) for s in node.strides],
            dtype=data_type_to_dace_typeclass(node.dtype),
            storage=node.storage.to_dace_storage(),
            transient=node.name not in non_transients,
            debuginfo=dace.DebugInfo(0),
        )

    def visit_SymbolDecl(
        self,
        node: dcir.SymbolDecl,
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        **kwargs,
    ) -> None:
        if node.name not in sdfg_ctx.sdfg.symbols:
            sdfg_ctx.sdfg.add_symbol(node.name, stype=data_type_to_dace_typeclass(node.dtype))

    def visit_NestedSDFG(
        self,
        node: dcir.NestedSDFG,
        *,
        sdfg_ctx: Optional[StencilComputationSDFGBuilder.SDFGContext] = None,
        node_ctx: Optional[StencilComputationSDFGBuilder.NodeContext] = None,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs,
    ) -> dace.nodes.NestedSDFG:
        sdfg = dace.SDFG(node.label)
        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg, state=sdfg.add_state(is_start_state=True)
        )

        for declaration in node.field_decls:
            self.visit(
                declaration,
                sdfg_ctx=inner_sdfg_ctx,
                non_transients={
                    memlet.connector for memlet in node.read_memlets + node.write_memlets
                },
                **kwargs,
            )

        self.visit(node.symbol_decls, sdfg_ctx=inner_sdfg_ctx, **kwargs)
        symbol_mapping = {decl.name: decl.to_dace_symbol() for decl in node.symbol_decls}

        # build access nodes and connect them to {input,output}_connectors of the nested SDFG
        # use an "inner node context" for this and supply this context to the visitor of
        # computation states (below)
        read_access_and_connectors: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
        for connector in node.input_connectors:
            read_access_and_connectors[connector] = (
                inner_sdfg_ctx.state.add_access(connector, debuginfo=dace.DebugInfo(0)),
                None,
            )

        write_access_and_connectors: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
        for connector in node.output_connectors:
            write_access_and_connectors[connector] = (
                inner_sdfg_ctx.state.add_access(connector, debuginfo=dace.DebugInfo(0)),
                None,
            )

        inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
            input_node_and_conns=read_access_and_connectors,
            output_node_and_conns=write_access_and_connectors,
        )

        for computation_state in node.states:
            self.visit(
                computation_state,
                sdfg_ctx=inner_sdfg_ctx,
                symtable=symtable,
                node_ctx=inner_node_ctx,
                **kwargs,
            )

        if sdfg_ctx is not None and node_ctx is not None:
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=sdfg_ctx.sdfg,
                inputs=node.input_connectors,
                outputs=node.output_connectors,
                symbol_mapping=symbol_mapping,
                debuginfo=dace.DebugInfo(0),
            )
            self.visit(
                node.read_memlets,
                scope_node=nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                symtable=symtable.parents,
                **kwargs,
            )
            self.visit(
                node.write_memlets,
                scope_node=nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                symtable=symtable.parents,
                **kwargs,
            )
            StencilComputationSDFGBuilder._add_empty_edges(
                nsdfg, nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )
        else:
            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs={memlet.connector for memlet in node.read_memlets},
                outputs={memlet.connector for memlet in node.write_memlets},
                symbol_mapping=symbol_mapping,
            )

        return nsdfg

    def _add_condition_evaluation_tasklet(
        self,
        tmp_condition_name: str,
        node: Union[dcir.Condition, dcir.WhileLoop],
        *,
        sdfg_ctx: StencilComputationSDFGBuilder.SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs,
    ) -> None:
        tmp_access = dcir.ScalarAccess(name=tmp_condition_name, dtype=common.DataType.BOOL)
        condition_tasklet = dcir.Tasklet(
            decls=[
                dcir.LocalScalarDecl(
                    name=tmp_access.name, dtype=tmp_access.dtype, loc=tmp_access.loc
                )
            ],
            stmts=[dcir.AssignStmt(left=tmp_access, right=node.condition, loc=tmp_access.loc)],
            read_memlets=[],
            write_memlets=[],
        )
        self.visit(
            condition_tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, symtable=symtable, **kwargs
        )
