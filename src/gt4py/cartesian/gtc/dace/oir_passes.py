# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from gt4py import eve
from gt4py.cartesian.gtc import oir


class FixIJCacheDimensions(eve.NodeTranslator):
    """IJCaches are translate 3d temporaries into 2d fields. Cache detection only detects
    caches and doesn't to do anything else. Here we 'fix' the dimension to be 2d for
    detected IJCaches. This will allow us to transparently generate IJCaches."""

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        for vertical_loop in node.vertical_loops:
            for ij_cache in filter(
                lambda cache: isinstance(cache, oir.IJCache), vertical_loop.caches
            ):
                for tmp in filter(lambda tmp: tmp.name == ij_cache.name, node.declarations):
                    tmp.dimensions = (True, True, False)
        return node
