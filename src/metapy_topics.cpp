/**
 * @file metapy_topics.cpp
 * @author Sean Massung
 */

#include <pybind11/pybind11.h>

#include "cpptoml.h"
#include "meta/learn/dataset.h"
#include "meta/logging/logger.h"
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
        .def("run", &topics::lda_model::run)
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
        .def("num_topics", &topics::lda_model::num_topics);

    py::class_<topics::lda_cvb, topics::lda_model>{m_topics, "LDACollapsedVB"}
        .def(py::init<const learn::dataset&, std::size_t, double, double>(),
             py::arg("docs"), py::arg("num_topics"), py::arg("alpha"),
             py::arg("beta"))
        .def("run", &topics::lda_cvb::run, py::arg("num_iters"),
             py::arg("convergence") = 1e-3);

    py::class_<topics::lda_gibbs, topics::lda_model>{m_topics, "LDAGibbs"}
        .def(py::init<const learn::dataset&, std::size_t, double, double>(),
             py::arg("docs"), py::arg("num_topics"), py::arg("alpha"),
             py::arg("beta"))
        .def("run", &topics::lda_gibbs::run, py::arg("num_iters"),
             py::arg("convergence") = 1e-6);

    py::class_<topics::parallel_lda_gibbs, topics::lda_gibbs>{
        m_topics, "LDAParallelGibbs"}
        .def(py::init<const learn::dataset&, std::size_t, double, double>(),
             py::arg("docs"), py::arg("num_topics"), py::arg("alpha"),
             py::arg("beta"));

    py::class_<topics::lda_scvb, topics::lda_model>{m_topics,
                                                    "LDAStochasticCVB"}
        .def(py::init<const learn::dataset&, std::size_t, double, double,
                      uint64_t>(),
             py::arg("docs"), py::arg("num_topics"), py::arg("alpha"),
             py::arg("beta"), py::arg("minibatch_size") = 100)
        .def("run", &topics::lda_scvb::run, py::arg("num_iters"),
             py::arg("convergence") = 0);

    m_topics
        .def("run_gibbs",
             [](const std::string& config_path, const std::string& out_prefix,
                std::size_t num_topics, double alpha, double beta,
                std::size_t num_iters, double convergence) {
                 run_lda<topics::lda_gibbs>(config_path, out_prefix, num_topics,
                                            alpha, beta, num_iters,
                                            convergence);
             },
             py::arg("config_path"), py::arg("out_prefix"),
             py::arg("num_topics"), py::arg("alpha") = 0.1,
             py::arg("beta") = 0.1, py::arg("num_iters") = 500,
             py::arg("convergence") = 1e-6)
        .def("run_cvb",
             [](const std::string& config_path, const std::string& out_prefix,
                std::size_t num_topics, double alpha, double beta,
                size_t num_iters, double convergence) {
                 run_lda<topics::lda_cvb>(config_path, out_prefix, num_topics,
                                          alpha, beta, num_iters, convergence);
             },
             py::arg("config_path"), py::arg("out_prefix"),
             py::arg("num_topics"), py::arg("alpha") = 0.1,
             py::arg("beta") = 0.1, py::arg("num_iters") = 500,
             py::arg("convergence") = 1e-6)
        .def("run_parallel_gibbs",
             [](const std::string& config_path, const std::string& out_prefix,
                std::size_t num_topics, double alpha, double beta,
                std::size_t num_iters, double convergence) {
                 run_lda<topics::parallel_lda_gibbs>(config_path, out_prefix,
                                                     num_topics, alpha, beta,
                                                     num_iters, convergence);
             },
             py::arg("config_path"), py::arg("out_prefix"),
             py::arg("num_topics"), py::arg("alpha") = 0.1,
             py::arg("beta") = 0.1, py::arg("num_iters") = 500,
             py::arg("convergence") = 1e-6)
        .def("run_scvb",
             [](const std::string& config_path, const std::string& out_prefix,
                std::size_t num_topics, double alpha, double beta,
                std::size_t num_iters, double convergence) {
                 run_lda<topics::lda_scvb>(config_path, out_prefix, num_topics,
                                           alpha, beta, num_iters, convergence);
             },
             py::arg("config_path"), py::arg("out_prefix"),
             py::arg("num_topics"), py::arg("alpha") = 0.1,
             py::arg("beta") = 0.1, py::arg("num_iters") = 500,
             py::arg("convergence") = 1e-6);
}
