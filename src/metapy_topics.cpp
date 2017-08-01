/**
 * @file metapy_topics.cpp
 * @author Sean Massung
 */

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "cpptoml.h"
#include "meta/learn/dataset.h"
#include "meta/logging/logger.h"
#include "meta/topics/bl_term_scorer.h"
#include "meta/topics/lda_cvb.h"
#include "meta/topics/lda_gibbs.h"
#include "meta/topics/lda_scvb.h"
#include "meta/topics/parallel_lda_gibbs.h"
#include "metapy_stats.h"
#include "metapy_topics.h"

namespace py = pybind11;
using namespace meta;

template <class Model>
void run_lda(const std::string& config_path, const std::string& out_prefix,
             std::size_t num_topics, double alpha, double beta,
             std::size_t num_iters, double convergence)
{
    logging::set_cerr_logging();
    auto cfg = cpptoml::parse_file(config_path);
    auto f_idx = index::make_index<index::forward_index>(*cfg);
    auto doc_list = f_idx->docs();
    learn::dataset docs{f_idx, doc_list.begin(), doc_list.end()};
    Model model{docs, num_topics, alpha, beta};
    model.run(num_iters, convergence);
    model.save(out_prefix);
}

void metapy_bind_topics(py::module& m)
{
    auto m_topics = m.def_submodule("topics");

    py::class_<topics::lda_model>{m_topics, "LDAModel"}
        .def("run",
             [](topics::lda_model& model, uint64_t num_iters,
                double convergence) {
                 py::gil_scoped_release release;
                 model.run(num_iters, convergence);
             })
        .def("save_doc_topic_distributions",
             [](const topics::lda_model& model, const std::string& filename) {
                 std::ofstream output{filename, std::ios::binary};
                 model.save_doc_topic_distributions(output);
             })
        .def("save_topic_term_distributions",
             [](const topics::lda_model& model, const std::string& filename) {
                 std::ofstream output{filename, std::ios::binary};
                 model.save_topic_term_distributions(output);
             })
        .def("save", &topics::lda_model::save)
        .def("compute_term_topic_probability",
             &topics::lda_model::compute_term_topic_probability)
        .def("compute_doc_topic_probability",
             &topics::lda_model::compute_doc_topic_probability)
        .def("topic_distribution",
             [](const topics::lda_model& model, doc_id doc) {
                 return py_multinomial{model.topic_distribution(doc)};
             })
        .def("term_distribution",
             [](const topics::lda_model& model, topic_id k) {
                 return py_multinomial{model.term_distribution(k)};
             })
        .def("num_topics", &topics::lda_model::num_topics);

    py::class_<topics::lda_cvb, topics::lda_model>{m_topics, "LDACollapsedVB"}
        .def(py::init<const learn::dataset&, std::size_t, double, double>(),
             py::keep_alive<0, 1>(), py::arg("docs"), py::arg("num_topics"),
             py::arg("alpha"), py::arg("beta"))
        .def("run",
             [](topics::lda_cvb& lda, uint64_t num_iters, double convergence) {
                 py::gil_scoped_release release;
                 lda.run(num_iters, convergence);
             },
             py::arg("num_iters"), py::arg("convergence") = 1e-3);

    py::class_<topics::lda_gibbs, topics::lda_model>{m_topics, "LDAGibbs"}
        .def(py::init<const learn::dataset&, std::size_t, double, double>(),
             py::keep_alive<0, 1>(), py::arg("docs"), py::arg("num_topics"),
             py::arg("alpha"), py::arg("beta"))
        .def(
            "run",
            [](topics::lda_gibbs& lda, uint64_t num_iters, double convergence) {
                py::gil_scoped_release release;
                lda.run(num_iters, convergence);
            },
            py::arg("num_iters"), py::arg("convergence") = 1e-6);

    py::class_<topics::parallel_lda_gibbs, topics::lda_gibbs>{
        m_topics, "LDAParallelGibbs"}
        .def(py::init<const learn::dataset&, std::size_t, double, double>(),
             py::keep_alive<0, 1>(), py::arg("docs"), py::arg("num_topics"),
             py::arg("alpha"), py::arg("beta"));

    py::class_<topics::lda_scvb, topics::lda_model>{m_topics,
                                                    "LDAStochasticCVB"}
        .def(py::init<const learn::dataset&, std::size_t, double, double,
                      uint64_t>(),
             py::keep_alive<0, 1>(), py::arg("docs"), py::arg("num_topics"),
             py::arg("alpha"), py::arg("beta"), py::arg("minibatch_size") = 100)
        .def("run",
             [](topics::lda_scvb& lda, uint64_t num_iters, double convergence) {
                 py::gil_scoped_release release;
                 lda.run(num_iters, convergence);
             },
             py::arg("num_iters"), py::arg("convergence") = 0);

    py::class_<topics::topic_model>{m_topics, "TopicModel"}
        .def("__init__",
             [](topics::topic_model& model, const std::string& prefix) {
                 py::gil_scoped_release release;

                 std::ifstream theta{prefix + ".theta.bin", std::ios::binary};

                 if (!theta)
                 {
                     throw topics::topic_model_exception{
                         "missing document topic probabilities file: " + prefix
                         + ".theta.bin"};
                 }

                 std::ifstream phi{prefix + ".phi.bin", std::ios::binary};
                 if (!phi)
                 {
                     throw topics::topic_model_exception{
                         "missing topic term probabilities file: " + prefix
                         + ".phi.bin"};
                 }

                 new (&model) topics::topic_model(theta, phi);
             })
        .def("top_k", [](const topics::topic_model& model, topic_id tid,
                         std::size_t k) { return model.top_k(tid, k); },
             py::arg("tid"), py::arg("k") = 10)
        .def("top_k",
             [](const topics::topic_model& model, topic_id tid, std::size_t k,
                std::function<double(topic_id, term_id)> scorer) {
                 return model.top_k(tid, k, scorer);
             },
             py::arg("tid"), py::arg("k") = 10, py::arg("scorer"))
        .def("top_k",
             [](const topics::topic_model& model, topic_id tid, std::size_t k,
                const topics::bl_term_scorer& scorer) {
                 return model.top_k(tid, k, scorer);
             },
             py::arg("tid"), py::arg("k") = 10, py::arg("scorer"))
        .def("topic_distribution",
             [](const topics::topic_model& self, doc_id did) {
                 return py_multinomial{self.topic_distribution(did)};
             })
        .def("term_distribution",
             [](const topics::topic_model& self, topic_id k) {
                 return py_multinomial{self.term_distribution(k)};
             })
        .def("term_probability", &topics::topic_model::term_probability)
        .def("topic_probability", &topics::topic_model::topic_probability)
        .def("num_topics", &topics::topic_model::num_topics)
        .def("num_words", &topics::topic_model::num_words);

    m_topics.def("load_topic_model", [](const std::string& config_path) {
        py::gil_scoped_release release;
        auto config = cpptoml::parse_file(config_path);
        return topics::load_topic_model(*config);
    });

    py::class_<topics::bl_term_scorer>{m_topics, "BLTermScorer"}
        .def(py::init<const topics::topic_model&>(), py::keep_alive<0, 1>())
        .def("__call__", &topics::bl_term_scorer::operator());
}
