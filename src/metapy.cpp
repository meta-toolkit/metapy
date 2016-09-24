/**
 * @file metapy.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy module and bindings for the MeTA API. It
 * does not attempt to be completely comprehensive at this time (though
 * that is an eventual goal), but it aims to provide at least enough of an
 * API surface so that interactive web demos can be made.
 */

#include "metapy_analyzers.h"
#include "metapy_embeddings.h"
#include "metapy_index.h"
#include "metapy_parser.h"
#include "metapy_sequence.h"

namespace py = pybind11;

PYBIND11_PLUGIN(metapy)
{
    py::module m{"metapy", "MeTA toolkit python bindings"};

    metapy_bind_index(m);
    metapy_bind_analyzers(m);
    metapy_bind_sequence(m);
    metapy_bind_parser(m);
    metapy_bind_embeddings(m);

    return m.ptr();
}
