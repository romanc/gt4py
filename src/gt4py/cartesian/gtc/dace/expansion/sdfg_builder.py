# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ChainMap, Dict, Optional, Set, Tuple, Union

import dace
import dace.data
import dace.library
import dace.subsets

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace.expansion.sdfg_context import SDFGContext
from gt4py.cartesian.gtc.dace.expansion.tasklet_codegen import TaskletCodegen
from gt4py.cartesian.gtc.dace.expansion.utils import get_dace_debuginfo
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gt4py.cartesian.gtc.dace.utils import make_dace_subset


class StencilComputationSDFGBuilder(eve.VisitorWithSymbolTableTrait):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]

    def visit_Memlet(
        self,
        node: dcir.Memlet,
        *,
        scope_node: dcir.ComputationNode,
        sdfg_ctx: SDFGContext,
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
        sdfg_ctx: SDFGContext,
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
        sdfg_ctx: SDFGContext,
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
        # update edge (from while_init to while_guard) here, now that we know
        # what to put in the assignment dict
        # edges = sdfg_ctx.sdfg.edges_between("while_init", "while_guard")
        # assert len(edges) == 1
        # edges[0].data.assignments.update({"loop_condition": "condition_name_as_exported_from_the_tasklet"})

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
        sdfg_ctx: SDFGContext,
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

        # pop "condition_guard" here

        # fetch the edge
        # update the assignments on the edge with the condition as exported from the tasklet

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
        sdfg_ctx: SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        scalar_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        code = TaskletCodegen.apply_codegen(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            sdfg_ctx=sdfg_ctx,
            symtable=symtable,
        )

        scalar_mapping = {} if scalar_mapping is None else scalar_mapping
        tasklet_inputs: Set[str] = set()
        tasklet_outputs: Set[str] = set()

        # general idea:
        # either
        #  - use `tasklet_outputs` below and write into node_ctx
        #  - keep names from LocalScalarDecl inside tasklet (we can't control these in general)
        #  - when feeding back into another tasklet, use the same connector name and keep the internal name again (as in code)
        #  - get rid of `scalar_mapping` again (and fix condition/while loops as outlined there)
        # or
        #  - check if (and if so how) we could leverage `symtable`
        #  - we'd need to "convert" locals to symbols defined in that mapping

        # merge write_memlets with writes of local scalar declarations (as defined by node.decls)
        for access_node in node.walk_values().if_isinstance(dcir.AssignStmt):
            target_name = access_node.left.name

            field_access = (
                len(
                    set(
                        [
                            memlet.connector
                            for memlet in [*node.write_memlets]
                            if memlet.connector == target_name
                        ]
                    )
                )
                > 0
            )

            if field_access:
                continue

            matches = [declaration for declaration in node.decls if declaration.name == target_name]

            if len(matches) > 1:
                raise RuntimeError(
                    "Found more than one matching declaration for '%s'" % target_name
                )

            if len(matches) > 0:
                tasklet_outputs.add(target_name)
                sdfg_ctx.sdfg.add_scalar(
                    scalar_mapping[target_name], dtype=data_type_to_dace_typeclass(matches[0].dtype), transient=True
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
                        [
                            memlet.connector
                            for memlet in [*node.read_memlets, *node.write_memlets]
                            if memlet.connector == read_name
                        ]
                    )
                )
                > 0
            )
            defined_symbol = False
            for symbol_map in symtable.maps:
                for symbol in symbol_map.keys():
                    if symbol == read_name:
                        defined_symbol = True

            if not locally_declared and not field_access and not defined_symbol:
                tasklet_inputs.add(read_name)

        inputs = set(memlet.connector for memlet in node.read_memlets).union(tasklet_inputs)
        outputs = set(memlet.connector for memlet in node.write_memlets).union(tasklet_outputs)

        tasklet = sdfg_ctx.state.add_tasklet(
            name=f"{sdfg_ctx.sdfg.label}_Tasklet",
            code=code,
            inputs=inputs,
            outputs=outputs,
            debuginfo=get_dace_debuginfo(node),
        )

        # add memlets for local scalars into / out of tasklet
        for connector in tasklet_outputs:
            output_name = scalar_mapping[connector]
            access_node = sdfg_ctx.state.add_access(output_name)
            write_memlet = dace.Memlet(data=output_name)
            sdfg_ctx.state.add_memlet_path(
                tasklet, access_node, src_conn=connector, memlet=write_memlet
            )
        for input_name in tasklet_inputs:
            access_node = sdfg_ctx.state.add_access(input_name)
            read_memlet = dace.Memlet(data=input_name)
            sdfg_ctx.state.add_memlet_path(
                access_node, tasklet, dst_conn=input_name, memlet=read_memlet
            )

        # Fill up node_ctx with access nodes here for later use when visiting Memlets
        # NOTE This is yet another sign that our abstractions (of dace) are bad
        for memlet in node.read_memlets:
            access_node = sdfg_ctx.state.add_access(memlet.field)
            node_ctx.input_node_and_conns[memlet.field] = (access_node, None)
        for memlet in node.write_memlets:
            access_node = sdfg_ctx.state.add_access(memlet.field)
            node_ctx.output_node_and_conns[memlet.field] = (access_node, None)

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
        sdfg_ctx: SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
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

        # TODO fill up node_ctx with access nodes here for later use when visiting Memlets
        # NOTE This is yet another sign that our abstractions (of dace) are bad
        for memlet in node.read_memlets:
            if memlet.field not in node_ctx.input_node_and_conns:
                access_node = sdfg_ctx.state.add_access(memlet.field)
                node_ctx.input_node_and_conns[memlet.field] = (access_node, None)
        for memlet in node.write_memlets:
            if memlet.field not in node_ctx.output_node_and_conns:
                access_node = sdfg_ctx.state.add_access(memlet.field)
                node_ctx.output_node_and_conns[memlet.field] = (access_node, None)

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
        sdfg_ctx: SDFGContext,
        **kwargs,
    ) -> None:
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        **kwargs,
    ) -> None:
        sdfg_ctx.add_state()
        for computation in node.computations:
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, **kwargs)

    def visit_FieldDecl(
        self,
        node: dcir.FieldDecl,
        *,
        sdfg_ctx: SDFGContext,
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
        sdfg_ctx: SDFGContext,
        **kwargs,
    ) -> None:
        if node.name not in sdfg_ctx.sdfg.symbols:
            sdfg_ctx.sdfg.add_symbol(node.name, stype=data_type_to_dace_typeclass(node.dtype))

    def visit_NestedSDFG(
        self,
        node: dcir.NestedSDFG,
        *,
        sdfg_ctx: Optional[SDFGContext] = None,
        node_ctx: Optional[StencilComputationSDFGBuilder.NodeContext] = None,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs,
    ) -> dace.nodes.NestedSDFG:
        sdfg = dace.SDFG(node.label)
        inner_sdfg_ctx = SDFGContext(
            sdfg=sdfg,
            state=sdfg.add_state(is_start_state=True, label="nSDFG_start"),
        )

        self.visit(
            node.field_decls,
            sdfg_ctx=inner_sdfg_ctx,
            non_transients={memlet.connector for memlet in node.read_memlets + node.write_memlets},
            **kwargs,
        )

        self.visit(node.symbol_decls, sdfg_ctx=inner_sdfg_ctx, **kwargs)
        symbol_mapping = {decl.name: decl.to_dace_symbol() for decl in node.symbol_decls}

        inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
            input_node_and_conns={},
            output_node_and_conns={},
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

            # Fill up node_ctx with access nodes here for later use when visiting Memlets
            # NOTE This is yet another sign that our abstractions (of dace) are bad
            for memlet in node.read_memlets:
                if memlet.field not in node_ctx.input_node_and_conns:
                    access_node = sdfg_ctx.state.add_access(memlet.field)
                    node_ctx.input_node_and_conns[memlet.field] = (access_node, None)
            for memlet in node.write_memlets:
                if memlet.field not in node_ctx.output_node_and_conns:
                    access_node = sdfg_ctx.state.add_access(memlet.field)
                    node_ctx.output_node_and_conns[memlet.field] = (access_node, None)

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
        sdfg_ctx: SDFGContext,
        node_ctx: StencilComputationSDFGBuilder.NodeContext,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs,
    ) -> None:
        local_name = "condition_expression"
        tmp_access = dcir.ScalarAccess(name=local_name, dtype=common.DataType.BOOL)
        condition_tasklet = dcir.Tasklet(
            decls=[
                dcir.LocalScalarDecl(
                    name=local_name, dtype=tmp_access.dtype, loc=tmp_access.loc
                )
            ],
            stmts=[dcir.AssignStmt(left=tmp_access, right=node.condition, loc=tmp_access.loc)],
            read_memlets=[],
            write_memlets=[],
        )
        self.visit(
            condition_tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, symtable=symtable,
            scalar_mapping={local_name: tmp_condition_name},  **kwargs
        )
