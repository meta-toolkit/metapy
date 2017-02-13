/**
 * @file metapy_sequence.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy.sequence submodule and creates bindings
 * for that part of the MeTA API.
 */

#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpptoml.h"

#include "meta/sequence/io/ptb_parser.h"
#include "meta/sequence/perceptron.h"
#include "meta/sequence/sequence.h"

namespace py = pybind11;
using namespace meta;

void metapy_bind_sequence(py::module& m)
{
    auto m_seq = m.def_submodule("sequence");

    py::class_<sequence::observation>{m_seq, "Observation"}
        .def(py::init<sequence::symbol_t, sequence::tag_t>())
        .def(py::init<sequence::symbol_t>())
        .def_property(
            "symbol",
            [](const sequence::observation& obs) { return obs.symbol(); },
            [](sequence::observation& obs, sequence::symbol_t sym) {
                obs.symbol(std::move(sym));
            })
        .def_property(
            "tag", [](const sequence::observation& obs) { return obs.tag(); },
            [](sequence::observation& obs, sequence::tag_t tag) {
                obs.tag(std::move(tag));
            })
        .def_property(
            "label",
            [](const sequence::observation& obs) { return obs.label(); },
            [](sequence::observation& obs, label_id lbl) { obs.label(lbl); })
        .def_property(
            "features",
            [](const sequence::observation& obs) { return obs.features(); },
            [](sequence::observation& obs,
               sequence::observation::feature_vector feats) {
                obs.features(std::move(feats));
            })
        .def("tagged", &sequence::observation::tagged);

    py::class_<sequence::sequence>{m_seq, "Sequence"}
        .def(py::init<>())
        .def("add_observation", &sequence::sequence::add_observation)
        .def("add_symbol", &sequence::sequence::add_symbol)
        .def("__getitem__",
             [](sequence::sequence& seq, sequence::sequence::size_type idx) {
                 if (idx >= seq.size())
                     throw py::index_error();
                 return seq[idx];
             })
        .def("__setitem__",
             [](sequence::sequence& seq, sequence::sequence::size_type idx,
                sequence::observation obs) {
                 if (idx >= seq.size())
                     throw py::index_error();
                 seq[idx] = std::move(obs);
             })
        .def("__len__", &sequence::sequence::size)
        .def("__iter__",
             [](const sequence::sequence& seq) {
                 return py::make_iterator(seq.begin(), seq.end());
             },
             py::keep_alive<0, 1>())
        .def("__str__",
             [](const sequence::sequence& seq) {
                 std::string res;
                 for (auto it = seq.begin(); it != seq.end();)
                 {
                     res += "(" + static_cast<std::string>(it->symbol()) + ", "
                            + (it->tagged()
                                   ? static_cast<std::string>(it->tag())
                                   : "???")
                            + ")";
                     if (++it != seq.end())
                         res += ", ";
                 }
                 return res;
             })
        .def("tagged", [](const sequence::sequence& seq) {
            std::vector<std::pair<std::string, std::string>> res(seq.size());
            std::transform(seq.begin(), seq.end(), res.begin(),
                           [](const sequence::observation& obs) {
                               return std::make_pair(
                                   obs.symbol(),
                                   obs.tagged()
                                       ? static_cast<std::string>(obs.tag())
                                       : "???");
                           });
            return res;
        });

    m_seq.def("extract_sequences", &sequence::extract_sequences);

    using sequence::perceptron;
    py::class_<perceptron> perc_tagger{m_seq, "PerceptronTagger"};

    py::class_<perceptron::training_options>{perc_tagger, "TrainingOptions"}
        .def(py::init<>())
        .def_readwrite("max_iterations",
                       &perceptron::training_options::max_iterations)
        .def_readwrite("seed", &perceptron::training_options::seed);

    perc_tagger.def(py::init<>())
        .def(py::init<const std::string&>())
        .def("tag", &perceptron::tag)
        .def("train", &perceptron::train)
        .def("save", &perceptron::save);
}
