/**
 * @file metapy_classify.cpp
 * @author Chase Geigle
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "cpptoml.h"
#include "meta/classify/binary_dataset_view.h"
#include "meta/classify/classifier/all.h"
#include "meta/learn/dataset.h"
#include "meta/learn/loss/loss_function_factory.h"
#include "meta/logging/logger.h"
#include "meta/util/iterator.h"
#include "metapy_classify.h"
#include "metapy_identifiers.h"
#include "metapy_learn.h"

namespace py = pybind11;
using namespace meta;

template <class ClassifierBase = classify::binary_classifier>
class py_binary_classifier : public ClassifierBase
{
  public:
    double predict(const learn::feature_vector& instance) const override
    {
        PYBIND11_OVERLOAD_PURE(double, ClassifierBase, predict, instance);
        return 0;
    }

    void save(std::ostream& /* os */) const override
    {
        throw std::runtime_error{
            "cannot serialize python-defined binary classifiers"};
    }
};

class py_online_binary_classifier
    : public py_binary_classifier<classify::online_binary_classifier>
{
  public:
    void train(dataset_view_type docs) override
    {
        PYBIND11_OVERLOAD_PURE(void, classify::online_binary_classifier, train,
                               docs);
    }

    void train_one(const feature_vector& doc, bool label) override
    {
        PYBIND11_OVERLOAD_PURE(void, classify::online_binary_classifier,
                               train_one, doc, label);
    }
};

void metapy_bind_classify(py::module& m)
{
    auto pydset = (py::object)m.attr("learn").attr("Dataset");
    auto pydset_view = (py::object)m.attr("learn").attr("DatasetView");
    auto m_classify = m.def_submodule("classify");

    py::class_<classify::binary_dataset>{m_classify, "BinaryDataset", pydset}
        .def(py::init<std::shared_ptr<index::forward_index>,
                      std::function<bool(doc_id)>>())
        .def("label", &classify::binary_dataset::label)
        .def("__getitem__",
             [](const classify::binary_dataset& bdset, py::slice slice) {
                 classify::binary_dataset_view bdv{bdset};
                 return make_sliced_dataset_view(bdv, slice);
             },
             py::keep_alive<0, 1>());

    py::class_<classify::binary_dataset_view>{m_classify, "BinaryDatasetView",
                                              pydset_view}
        .def(py::init<const classify::binary_dataset&>(),
             py::keep_alive<0, 1>())
        .def("__getitem__",
             [](const classify::binary_dataset_view& bdv, py::slice slice) {
                 return make_sliced_dataset_view(bdv, slice);
             },
             py::keep_alive<0, 1>());

    py::implicitly_convertible<classify::binary_dataset,
                               classify::binary_dataset_view>();

    py::class_<classify::binary_classifier, py_binary_classifier<>> pybincls{
        m_classify, "BinaryClassifier"};
    pybincls.def("classify", &classify::binary_classifier::classify)
        .def("predict", &classify::binary_classifier::predict);

    py::class_<classify::online_binary_classifier, py_online_binary_classifier>
        py_online_bincls{m_classify, "OnlineBinaryClassifier", pybincls};
    py_online_bincls.def("train", &classify::online_binary_classifier::train)
        .def("train_one", &classify::online_binary_classifier::train_one);

    py::class_<classify::sgd>{m_classify, "SGD", py_online_bincls}
        .def_property_readonly_static(
            "id",
            [](py::object /* self */) { return classify::sgd::id.to_string(); })
        .def_readonly_static("default_gamma", &classify::sgd::default_gamma)
        .def_readonly_static("default_max_iter",
                             &classify::sgd::default_max_iter)
        .def("__init__",
             [](classify::sgd& cls, classify::binary_dataset_view training,
                const std::string& loss_id,
                learn::sgd_model::options_type options, double gamma,
                std::size_t max_iter, bool calibrate) {

                 new (&cls)
                     classify::sgd(std::move(training),
                                   learn::loss::make_loss_function(loss_id),
                                   options, gamma, max_iter, calibrate);
             },
             py::arg("training"), py::arg("loss_id"),
             py::arg("options") = learn::sgd_model::options_type{},
             py::arg("gamma") = classify::sgd::default_gamma,
             py::arg("max_iter") = classify::sgd::default_max_iter,
             py::arg("calibrate") = false);
}
