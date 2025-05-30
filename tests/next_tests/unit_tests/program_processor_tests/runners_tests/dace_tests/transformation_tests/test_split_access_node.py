# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import copy
import numpy as np

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _perform_test(
    sdfg: dace.SDFG,
    explected_applies: int,
    removed_transients: set[str] | None = None,
) -> None:
    ref = {
        name: np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype())
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    res = copy.deepcopy(ref)

    if removed_transients is not None:
        assert removed_transients.issubset(
            name for name, desc in sdfg.arrays.items() if desc.transient
        )

    if explected_applies != 0:
        util.compile_and_run_sdfg(sdfg, **ref)

    nb_apply = gtx_transformations.gt_split_access_nodes(
        sdfg=sdfg,
        validate=True,
        validate_all=True,
    )
    assert nb_apply == explected_applies

    if explected_applies == 0:
        return

    util.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())

    if removed_transients is not None:
        assert all(
            name not in removed_transients for name, desc in sdfg.arrays.items() if desc.transient
        )


def test_map_producer_ac_consumer():
    """The data is generated by a Map and then consumed by an AccessNode."""
    sdfg = dace.SDFG(util.unique_name("map_producer_ac_consumer"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcd":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=(name == "t"),
        )
    t = state.add_access("t")

    state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:5"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("t[__i + 10]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_nedge(t, state.add_access("b"), dace.Memlet("t[10:15] -> [3:8]"))

    state.add_nedge(state.add_access("c"), t, dace.Memlet("c[0:10] -> [0:10]"))
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[0:10] -> [3:13]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def test_map_producer_map_consumer():
    """The data is generated by a Map and then consumed by another Map."""
    sdfg = dace.SDFG(util.unique_name("map_producer_map_consumer"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcd":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=(name == "t"),
        )
    t = state.add_access("t")

    state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:5"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("t[__i + 10]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "10:15"},
        inputs={"__in": dace.Memlet("t[__i]")},
        code="__out = __in + 13.0",
        outputs={"__out": dace.Memlet("b[__i - 7]")},
        input_nodes={t},
        external_edges=True,
    )

    state.add_nedge(state.add_access("c"), t, dace.Memlet("c[0:10] -> [0:10]"))
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[0:10] -> [3:13]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def test_ac_producer_ac_consumer():
    sdfg = dace.SDFG(util.unique_name("ac_producer_ac_consumer"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcd":
        sdfg.add_array(
            name,
            shape=(40,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    t = state.add_access("t")

    state.add_nedge(state.add_access("a"), t, dace.Memlet("a[11:21] -> [5:15]"))
    state.add_nedge(state.add_access("b"), t, dace.Memlet("b[28:38] -> [20:30]"))

    state.add_nedge(t, state.add_access("c"), dace.Memlet("t[5:15] -> [0:10]"))
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[20:30] -> [22:32]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def test_ac_producer_map_consumer():
    sdfg = dace.SDFG(util.unique_name("ac_producer_map_consumer"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcd":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=(name == "t"),
        )
    t = state.add_access("t")

    state.add_nedge(state.add_access("a"), t, dace.Memlet("a[1:11] -> [0:10]"))
    state.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "5:10"},
        inputs={"__in": dace.Memlet("t[__i]")},
        code="__out = __in + 13.0",
        outputs={"__out": dace.Memlet("b[__i + 7]")},
        input_nodes={t},
        external_edges=True,
    )

    state.add_nedge(state.add_access("c"), t, dace.Memlet("c[0:10] -> [10:20]"))
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[10:20] -> [3:13]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def test_simple_splitable_ac_source_not_full_consume():
    """Similar to `test_simple_splitable_ac_source_full_consume`, but one consumer
    does not fully consumer what is produced.
    """
    sdfg = dace.SDFG(util.unique_name("simple_splitable_ac_source_not_full_consume"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcde":
        sdfg.add_array(
            name,
            shape=(40,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    t = state.add_access("t")

    state.add_nedge(state.add_access("a"), t, dace.Memlet("a[11:21] -> [5:15]"))
    state.add_nedge(state.add_access("b"), t, dace.Memlet("b[28:38] -> [20:30]"))

    state.add_nedge(t, state.add_access("c"), dace.Memlet("t[7:12] -> [1:6]"))
    state.add_nedge(t, state.add_access("e"), dace.Memlet("t[5:15] -> [8:18]"))
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[20:30] -> [22:32]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def test_simple_splitable_ac_source_multiple_consumer():
    """Similar to `test_simple_splitable_ac_source_not_full_consume`, but there are
    multiple consumer, per producer.
    """
    sdfg = dace.SDFG(util.unique_name("simple_splitable_ac_source_multiple_consumer"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcd":
        sdfg.add_array(
            name,
            shape=(40,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    t = state.add_access("t")

    state.add_nedge(state.add_access("a"), t, dace.Memlet("a[11:21] -> [5:15]"))
    state.add_nedge(state.add_access("b"), t, dace.Memlet("b[28:38] -> [20:30]"))

    state.add_nedge(
        t,
        state.add_access("c"),
        # Only a subset is consumed.
        dace.Memlet("t[7:12] -> [1:6]"),
    )
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[20:30] -> [22:32]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def _make_transient_producer_sdfg(
    partial_read: bool,
) -> dace.SDFG:
    """Data is generated by a transient.

    The read from the intermediate transient is can either be partial or full
    depending on the value of `partial_read`.
    """
    sdfg = dace.SDFG(
        util.unique_name("partial_ac_read" + ("_partial_read" if partial_read else "_full_read"))
    )
    state = sdfg.add_state(is_start_block=True)
    for name in ["a", "b", "c", "d", "t1", "t2"]:
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=name.startswith("t"),
        )
    t1 = state.add_access("t1")
    t2 = state.add_access("t2")

    state.add_nedge(state.add_access("a"), t1, dace.Memlet("t1[0:20] -> [0:20]"))
    state.add_nedge(t1, t2, dace.Memlet("t1[0:10] -> [0:10]"))

    if partial_read:
        state.add_nedge(t2, state.add_access("b"), dace.Memlet("t2[0:9] -> [0:9]"))
    else:
        state.add_nedge(t2, state.add_access("b"), dace.Memlet("t2[0:10] -> [0:10]"))

    state.add_nedge(state.add_access("c"), t2, dace.Memlet("c[10:20] -> [10:20]"))
    state.add_nedge(t2, state.add_access("d"), dace.Memlet("t2[10:20] -> [10:20]"))
    sdfg.validate()

    return sdfg


def test_transient_producer_full_read():
    # Because the read is performed in full, the transformation applies.
    sdfg = _make_transient_producer_sdfg(partial_read=False)
    _perform_test(sdfg, explected_applies=1, removed_transients={"t2"})


def test_transient_producer_partial_read():
    # Because the read from the intermediate is only partial the transformation
    #  does not apply. This might change in newer versions.
    sdfg = _make_transient_producer_sdfg(partial_read=True)
    _perform_test(sdfg, explected_applies=0)


def test_overlapping_consume_ac_source():
    """There are 2 producers, but only one consumer that needs both producers."""
    sdfg = dace.SDFG(util.unique_name("overlapping_consume_ac_source"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abtc":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=(name == "t"),
        )

    t = state.add_access("t")

    state.add_nedge(state.add_access("a"), t, dace.Memlet("a[1:11] -> [0:10]"))
    state.add_nedge(state.add_access("b"), t, dace.Memlet("b[2:12] -> [10:20]"))
    state.add_nedge(t, state.add_access("c"), dace.Memlet("t[0:20] -> [0:20]"))
    sdfg.validate()

    _perform_test(sdfg, explected_applies=0)


def _make_map_producer_multiple_consumer(
    partial_read: bool,
) -> dace.SDFG:
    """Creates an SDFG with a Map producer and multiple consumer.

    The Map generates a certain amount of data. That is fully read by another Map.
    The same data is also read by an AccessNode, depending on the value of
    `partial_read` it will either read the full data for only parts of it.
    """
    sdfg = dace.SDFG(
        util.unique_name(
            "map_producer_map_consumer" + ("_partial_read" if partial_read else "_full_read")
        )
    )
    state = sdfg.add_state(is_start_block=True)

    for name in "abtcde":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=(name == "t"),
        )
    t = state.add_access("t")

    state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:5"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("t[__i + 10]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "10:15"},
        inputs={"__in": dace.Memlet("t[__i]")},
        code="__out = __in + 13.0",
        outputs={"__out": dace.Memlet("b[__i - 7]")},
        input_nodes={t},
        external_edges=True,
    )

    if partial_read:
        state.add_nedge(t, state.add_access("e"), dace.Memlet("t[10:14] -> [3:7]"))
    else:
        state.add_nedge(t, state.add_access("e"), dace.Memlet("t[10:15] -> [3:8]"))

    state.add_nedge(state.add_access("c"), t, dace.Memlet("c[0:10] -> [0:10]"))
    state.add_nedge(t, state.add_access("d"), dace.Memlet("t[0:10] -> [3:13]"))
    sdfg.validate()
    return sdfg


def test_map_producer_multi_consumer_fullread():
    # Because the data is fully read that transformation will apply.
    sdfg = _make_map_producer_multiple_consumer(partial_read=False)
    _perform_test(sdfg, explected_applies=1, removed_transients={"t"})


def test_map_producer_multi_consumer_partialread():
    # Because the data is only partially read, the transformation will not apply.
    #  This might change in the future.
    sdfg = _make_map_producer_multiple_consumer(partial_read=True)
    _perform_test(sdfg, explected_applies=0)
