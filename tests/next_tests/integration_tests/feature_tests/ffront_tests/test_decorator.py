# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(dropd): Remove as soon as `gt4py.next.ffront.decorator` is type checked.
import unittest.mock as mock
import dataclasses

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next import metrics, allocators as next_allocators
from gt4py.next.program_processors.runners import gtfn
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case, cartesian_case_no_backend
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_program_gtir_regression(cartesian_case):
    @gtx.field_operator(backend=None)
    def testee_op(a: cases.IField) -> cases.IField:
        return a

    @gtx.program(backend=None)
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, out=out)

    assert isinstance(testee.gtir, itir.Program)
    assert isinstance(testee.with_backend(cartesian_case.backend).gtir, itir.Program)


def test_frozen(cartesian_case):
    if cartesian_case.backend is None:
        pytest.skip("Frozen Program with embedded execution is not possible.")

    @gtx.field_operator
    def testee_op(a: cases.IField) -> cases.IField:
        return a

    @gtx.program(backend=cartesian_case.backend, frozen=True)
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, out=out)

    assert isinstance(testee, gtx.ffront.decorator.FrozenProgram)

    # first run should JIT compile
    args_1, kwargs_1 = cases.get_default_data(cartesian_case, testee)
    testee(*args_1, offset_provider=cartesian_case.offset_provider, **kwargs_1)

    # _compiled_program should be set after JIT compiling
    args_2, kwargs_2 = cases.get_default_data(cartesian_case, testee)
    testee._compiled_program(*args_2, offset_provider=cartesian_case.offset_provider, **kwargs_2)

    # and give expected results
    assert np.allclose(kwargs_2["out"].ndarray, args_2[0].ndarray)

    # with_backend returns a new instance, which is frozen but not compiled yet
    assert testee.with_backend(cartesian_case.backend)._compiled_program is None

    # with_grid_type returns a new instance, which is frozen but not compiled yet
    assert testee.with_grid_type(cartesian_case.grid_type)._compiled_program is None


@pytest.mark.parametrize(
    "metrics_level,expected_names",
    [
        (metrics.DISABLED, ()),
        (metrics.PERFORMANCE, ("compute",)),
        (metrics.ALL, ("compute", "total")),
    ],
)
def test_collect_metrics(cartesian_case_no_backend, metrics_level, expected_names):
    cartesian_case = dataclasses.replace(
        cartesian_case_no_backend,
        backend=gtfn.run_gtfn,
        allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    )

    @gtx.field_operator
    def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
        return a + b

    @gtx.program(backend=None)
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, a, out=out)

    with mock.patch("gt4py.next.config.COLLECT_METRICS_LEVEL", metrics_level):
        testee = testee.with_backend(cartesian_case.backend).with_grid_type(
            cartesian_case.grid_type
        )
        args, kwargs = cases.get_default_data(cartesian_case, testee)
        testee(*args, offset_provider=cartesian_case.offset_provider, **kwargs)

    assert set(metrics.program_metrics.metric_names) == set(expected_names)
