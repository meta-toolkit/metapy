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
