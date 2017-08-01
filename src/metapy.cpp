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
#include "metapy_classify.h"
#include "metapy_embeddings.h"
#include "metapy_index.h"
#include "metapy_learn.h"
#include "metapy_parser.h"
#include "metapy_sequence.h"
#include "metapy_stats.h"
#include "metapy_topics.h"

#include "meta/logging/logger.h"
#include "meta/parser/analyzers/tree_analyzer.h"
#include "meta/sequence/analyzers/ngram_pos_analyzer.h"

namespace py = pybind11;

PYBIND11_PLUGIN(metapy)
{
    py::module m{"metapy", "MeTA toolkit python bindings"};

    meta::sequence::register_analyzers();
    meta::parser::register_analyzers();

    metapy_bind_index(m);
    metapy_bind_analyzers(m);
    metapy_bind_learn(m);
    metapy_bind_classify(m);
    metapy_bind_sequence(m);
    metapy_bind_parser(m);
    metapy_bind_embeddings(m);
    metapy_bind_stats(m);
    metapy_bind_topics(m);

    // printing::progress makes this really difficult to reason about.
    // Progress updating occurs from a separate thread. This is fine,
    // except that we need to use the Python stderr here instead of the
    // usual std::cerr. In order to do that, we need the GIL. We run into
    // problems when the current thread holds the GIL and then the progress
    // thread attempts to acquire it. So, **any function that uses progress
    // reporting must release the GIL before being invoked**!
    m.def("log_to_stderr", []() {
        // separate logging for progress output
        meta::logging::add_sink(
            {[](const std::string& line) {
                 py::gil_scoped_acquire gil;
                 py::module::import("sys").attr("stderr").attr("write")(line);
             },
             []() {},
             [](const meta::logging::logger::log_line& ll) {
                 return ll.severity()
                        == meta::logging::logger::severity_level::progress;
             },
             [](const meta::logging::logger::log_line& ll) {
                 return " " + ll.str();
             }});

        meta::logging::add_sink(
            {[](const std::string& line) {
                 py::gil_scoped_acquire gil;
                 py::module::import("sys").attr("stderr").attr("write")(line);
             },
             []() {}, meta::logging::logger::severity_level::trace});
    });

    return m.ptr();
}
