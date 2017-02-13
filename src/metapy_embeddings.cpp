/**
 * @file metapy_parser.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy.parser submodule and creates bindings for
 * that part of the MeTA API.
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpptoml.h"
#include "meta/embeddings/word_embeddings.h"
#include "metapy_embeddings.h"

namespace py = pybind11;
using namespace meta;

void metapy_bind_embeddings(py::module& m)
{
    auto m_emb = m.def_submodule("embeddings");

    using namespace py::literals;

    py::class_<embeddings::word_embeddings>{m_emb, "WordEmbeddings"}
        .def("at",
             [](embeddings::word_embeddings& self, const std::string& term) {
                 auto emb = self.at(term);

                 return py::make_tuple(emb.tid,
                                       py::array(emb.v.size(), emb.v.begin()));
             })
        .def("term", [](embeddings::word_embeddings& self,
                        std::size_t tid) { return self.term(tid).to_string(); })
        .def("top_k",
             [](embeddings::word_embeddings& self,
                py::array_t<double, py::array::c_style | py::array::forcecast>
                    query,
                std::size_t k) {
                 util::array_view<const double> avquery{query.data(),
                                                        query.size()};
                 auto scores = self.top_k(avquery, k);

                 std::vector<py::tuple> result;
                 result.reserve(scores.size());

                 std::transform(
                     scores.begin(), scores.end(), std::back_inserter(result),
                     [](const embeddings::scored_embedding& se) {
                         return py::make_tuple(
                             se.e.tid, py::array(se.e.v.size(), se.e.v.begin()),
                             se.score);
                     });
                 return result;
             },
             "query"_a, "k"_a = 100)
        .def("vector_size", &embeddings::word_embeddings::vector_size);

    m_emb.def("load_embeddings", [](const std::string& filename) {
        auto config = cpptoml::parse_file(filename);
        auto embed_cfg = config->get_table("embeddings");
        if (!embed_cfg)
            throw embeddings::word_embeddings_exception{
                "missing [embeddings] configuration in " + filename};

        return embeddings::load_embeddings(*embed_cfg);
    });
}
