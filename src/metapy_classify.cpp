/**
 * @file metapy_classify.cpp
 * @author Chase Geigle
 */

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpptoml.h"
#include "meta/classify/binary_dataset_view.h"
#include "meta/classify/classifier/all.h"
#include "meta/classify/kernel/all.h"
#include "meta/index/ranker/ranker_factory.h"
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

template <class ClassifierBase = classify::classifier>
class py_classifier : public ClassifierBase
{
  public:
    class_label classify(const learn::feature_vector& instance) const override
    {
        PYBIND11_OVERLOAD_PURE(class_label, ClassifierBase, classify, instance);
        return "[none]"_cl;
    }

    void save(std::ostream& /* os */) const override
    {
        throw std::runtime_error{
            "cannot serialize python-defined multiclass classifiers"};
    }
};

class py_online_classifier : public py_classifier<classify::online_classifier>
{
  public:
    void train(dataset_view_type docs) override
    {
        PYBIND11_OVERLOAD_PURE(void, classify::online_classifier, train, docs);
    }

    void train_one(const feature_vector& doc, const class_label& lbl) override
    {
        PYBIND11_OVERLOAD_PURE(void, classify::online_classifier, train_one,
                               doc, lbl);
    }
};

class py_kernel : public classify::kernel::kernel
{
  public:
    double operator()(const learn::feature_vector& first,
                      const learn::feature_vector& second) const override
    {
        PYBIND11_OVERLOAD_PURE_NAME(double, classify::kernel::kernel,
                                    "__call__", operator(), first, second);
        return 0;
    }

    void save(std::ostream& /* os */) const override
    {
        throw std::runtime_error{"cannot serialize python-defined kernels"};
    }
};

/**
 * This class holds a binary_classifier that was created by invoking
 * Python code.
 *
 * We need to be able to supply a function to the ensemble methods (e.g.
 * one_vs_all) that creates a std::unique_ptr<binary_classifier> from a
 * binary_dataset_view. We can't get std::unique_ptrs from Python code
 * directly. Instead, we grab a reference to the py::object that Python
 * created for us, and make a unique_ptr to this class that contains it and
 * just forwards the calls to the classifier by converting that py::object
 * to a binary_classifier reference.
 */
class cpp_created_py_binary_classifier
    : public classify::online_binary_classifier
{
  public:
    cpp_created_py_binary_classifier(py::object cls) : cls_{cls}
    {
        // nothing
    }

    double predict(const learn::feature_vector& instance) const override
    {
        return cls_.cast<classify::binary_classifier&>().predict(instance);
    }

    void save(std::ostream& os) const override
    {
        cls_.cast<classify::binary_classifier&>().save(os);
    }

    void train(classify::binary_dataset_view bdv) override
    {
        cls_.cast<classify::online_binary_classifier&>().train(bdv);
    }

    void train_one(const learn::feature_vector& instance, bool label) override
    {
        cls_.cast<classify::online_binary_classifier&>().train_one(instance,
                                                                   label);
    }

  private:
    py::object cls_;
};

void metapy_bind_classify(py::module& m)
{
    auto pydset = (py::object)m.attr("learn").attr("Dataset");
    auto pydset_view = (py::object)m.attr("learn").attr("DatasetView");
    auto m_classify = m.def_submodule("classify");

    // binary datasets/views
    py::class_<classify::binary_dataset>{m_classify, "BinaryDataset", pydset}
        .def("__init__",
             [](classify::binary_dataset& dset,
                const std::shared_ptr<index::forward_index>& fidx,
                std::function<bool(doc_id)> labeler) {
                 py::gil_scoped_release release;
                 new (&dset) classify::binary_dataset(fidx, labeler);
             })
        .def("__init__",
             [](classify::binary_dataset& dset,
                const std::shared_ptr<index::forward_index>& fidx,
                const std::vector<doc_id>& docs,
                std::function<bool(doc_id)> labeler) {
                 py::gil_scoped_release release;
                 new (&dset) classify::binary_dataset(fidx, docs, labeler);
             })
        .def("__init__",
             [](classify::binary_dataset& dset, py::list& data,
                std::size_t total_features, py::function& featurizer,
                py::function& labeler) {
                 new (&dset) classify::binary_dataset(
                     data.begin(), data.end(), total_features,
                     [&](py::handle obj) {
                         return py::cast<learn::feature_vector>(
                             featurizer(obj));
                     },
                     [&](py::handle obj) {
                         return py::cast<bool>(labeler(obj));
                     });
             })
        .def("label", &classify::binary_dataset::label)
        .def("__getitem__",
             [](const classify::binary_dataset& dset, int64_t offset) {
                 std::size_t idx = offset >= 0
                                       ? static_cast<std::size_t>(offset)
                                       : dset.size() + offset;
                 if (idx >= dset.size())
                     throw py::index_error();
                 return *(dset.begin() + idx);
             })
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
             [](const classify::binary_dataset_view& dv, int64_t offset) {
                 std::size_t idx = offset >= 0
                                       ? static_cast<std::size_t>(offset)
                                       : dv.size() + offset;
                 if (idx >= dv.size())
                     throw py::index_error();
                 return *(dv.begin() + idx);
             })
        .def("__getitem__",
             [](const classify::binary_dataset_view& bdv, py::slice slice) {
                 return make_sliced_dataset_view(bdv, slice);
             },
             py::keep_alive<0, 1>());

    py::implicitly_convertible<classify::binary_dataset,
                               classify::binary_dataset_view>();

    // multiclass datasets/views
    py::class_<classify::multiclass_dataset>{m_classify, "MulticlassDataset",
                                             pydset}
        .def("__init__",
             [](classify::multiclass_dataset& dset,
                const std::shared_ptr<index::forward_index>& fidx) {
                 py::gil_scoped_release release;
                 new (&dset) classify::multiclass_dataset(fidx);
             })
        .def("__init__",
             [](classify::multiclass_dataset& dset,
                const std::shared_ptr<index::forward_index>& fidx,
                const std::vector<doc_id>& docs) {
                 py::gil_scoped_release release;
                 new (&dset) classify::multiclass_dataset(fidx, docs);
             })
        .def("__init__",
             [](classify::multiclass_dataset& dset, py::list& data,
                std::size_t total_features, py::function& featurizer,
                py::function& labeler) {
                 new (&dset) classify::multiclass_dataset(
                     data.begin(), data.end(), total_features,
                     [&](py::handle obj) {
                         return py::cast<learn::feature_vector>(
                             featurizer(obj));
                     },
                     [&](py::handle obj) {
                         return py::cast<class_label>(labeler(obj));
                     });
             })
        .def("label",
             [](const classify::multiclass_dataset& dset,
                const learn::instance& inst) { return dset.label(inst); })
        .def("total_labels", &classify::multiclass_dataset::total_labels)
        .def("label_id_for", &classify::multiclass_dataset::label_id_for)
        .def("label_for", &classify::multiclass_dataset::label_for)
        .def("__getitem__",
             [](const classify::multiclass_dataset& dset, int64_t offset) {
                 std::size_t idx = offset >= 0
                                       ? static_cast<std::size_t>(offset)
                                       : dset.size() + offset;
                 if (idx >= dset.size())
                     throw py::index_error();
                 return *(dset.begin() + idx);
             })
        .def("__getitem__",
             [](const classify::multiclass_dataset& dset, py::slice slice) {
                 classify::multiclass_dataset_view mdv{dset};
                 return make_sliced_dataset_view(mdv, slice);
             },
             py::keep_alive<0, 1>());

    py::class_<classify::multiclass_dataset_view>{
        m_classify, "MulticlassDatasetView", pydset_view}
        .def(py::init<const classify::multiclass_dataset&>(),
             py::keep_alive<0, 1>())
        .def("__getitem__",
             [](const classify::multiclass_dataset_view& dv, int64_t offset) {
                 std::size_t idx = offset >= 0
                                       ? static_cast<std::size_t>(offset)
                                       : dv.size() + offset;
                 if (idx >= dv.size())
                     throw py::index_error();
                 return *(dv.begin() + idx);
             })
        .def("__getitem__",
             [](const classify::multiclass_dataset_view& mdv, py::slice slice) {
                 return make_sliced_dataset_view(mdv, slice);
             },
             py::keep_alive<0, 1>())
        .def("create_even_split",
             [](const classify::multiclass_dataset_view& mdv) {
                 return mdv.create_even_split();
             },
             py::keep_alive<0, 1>());

    py::implicitly_convertible<classify::multiclass_dataset,
                               classify::multiclass_dataset_view>();

    // confusion matrix
    py::class_<classify::confusion_matrix>{m_classify, "ConfusionMatrix"}
        .def(py::init<>())
        .def("add", &classify::confusion_matrix::add, py::arg("predicted"),
             py::arg("actual"), py::arg("num_times") = 1)
        .def("add_fold_accuracy",
             &classify::confusion_matrix::add_fold_accuracy)
        .def("fold_accuracy", &classify::confusion_matrix::fold_accuracy)
        .def("print_stats",
             [](const classify::confusion_matrix& matrix) {
                 std::stringstream ss;
                 matrix.print_stats(ss);
                 py::print(ss.str());
             })
        .def("__str__",
             [](const classify::confusion_matrix& matrix) {
                 std::stringstream ss;
                 matrix.print(ss);
                 return ss.str();
             })
        .def("print",
             [](py::object self) { py::print(self.attr("__str__")()); })
        .def("print_result_pairs",
             [](const classify::confusion_matrix& matrix) {
                 std::stringstream ss;
                 matrix.print_result_pairs(ss);
                 py::print(ss.str());
             })
        .def("predictions", &classify::confusion_matrix::predictions)
        .def("accuracy", &classify::confusion_matrix::accuracy)
        .def("f1_score",
             [](const classify::confusion_matrix& matrix) {
                 return matrix.f1_score();
             })
        .def("f1_score",
             [](const classify::confusion_matrix& matrix,
                const class_label& lbl) { return matrix.f1_score(lbl); })
        .def("precision",
             [](const classify::confusion_matrix& matrix) {
                 return matrix.precision();
             })
        .def("precision",
             [](const classify::confusion_matrix& matrix,
                const class_label& lbl) { return matrix.precision(lbl); })
        .def("recall",
             [](const classify::confusion_matrix& matrix) {
                 return matrix.recall();
             })
        .def("recall",
             [](const classify::confusion_matrix& matrix,
                const class_label& lbl) { return matrix.recall(lbl); })
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def_static("mcnemar_significant",
                    &classify::confusion_matrix::mcnemar_significant);

    // kernels
    auto m_kernel = m_classify.def_submodule("kernel");
    py::class_<classify::kernel::kernel, py_kernel> pykernel{m_classify,
                                                             "Kernel"};
    pykernel.def("__call__", &classify::kernel::kernel::operator());

    py::class_<classify::kernel::polynomial>{m_kernel, "Polynomial", pykernel}
        .def(py::init<uint8_t, double>(),
             py::arg("power") = classify::kernel::polynomial::default_power,
             py::arg("c") = classify::kernel::polynomial::default_c)
        .def_property_readonly_static(
            "id",
            [](py::object /* self */) {
                return classify::kernel::polynomial::id.to_string();
            })
        .def_readonly_static("default_power",
                             &classify::kernel::polynomial::default_power)
        .def_readonly_static("default_c",
                             &classify::kernel::polynomial::default_c);

    py::class_<classify::kernel::radial_basis>{m_kernel, "RadialBasis",
                                               pykernel}
        .def(py::init<double>(), py::arg("gamma"))
        .def_property_readonly_static("id", [](py::object /* self */) {
            return classify::kernel::radial_basis::id.to_string();
        });

    py::class_<classify::kernel::sigmoid>{m_kernel, "Sigmoid", pykernel}
        .def(py::init<double, double>(), py::arg("alpha"), py::arg("c"))
        .def_property_readonly_static("id", [](py::object /* self */) {
            return classify::kernel::sigmoid::id.to_string();
        });

    // binary classifiers
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
                 // release the GIL before training the classifier; this
                 // allows other threads inside an ensemble method to train
                 // simultaneiously
                 py::gil_scoped_release rel;
                 new (&cls) classify::sgd(
                     training, learn::loss::make_loss_function(loss_id),
                     options, gamma, max_iter, calibrate);
             },
             py::arg("training"), py::arg("loss_id"),
             py::arg("options") = learn::sgd_model::options_type{},
             py::arg("gamma") = classify::sgd::default_gamma,
             py::arg("max_iter") = classify::sgd::default_max_iter,
             py::arg("calibrate") = true);

    // multiclass classifiers
    py::class_<classify::classifier, py_classifier<>> pycls{m_classify,
                                                            "Classifier"};
    pycls.def("classify", &classify::classifier::classify)
        .def("test", &classify::classifier::test);

    py::class_<classify::online_classifier, py_online_classifier> py_online_cls{
        m_classify, "OnlineClassifier", pycls};
    py_online_cls.def("train", &classify::online_classifier::train)
        .def("train_one", &classify::online_classifier::train_one);

    py::class_<classify::dual_perceptron>{m_classify, "DualPerceptron", pycls}
        .def("__init__",
             [](classify::dual_perceptron& cls,
                classify::multiclass_dataset_view training,
                const classify::kernel::kernel& kernel, double alpha,
                double gamma, double bias, uint64_t max_iter) {
                 std::stringstream ss;
                 kernel.save(ss);

                 new (&cls) classify::dual_perceptron(
                     std::move(training), classify::kernel::load_kernel(ss),
                     alpha, gamma, bias, max_iter);
             },
             py::arg("training"), py::arg("kernel"),
             py::arg("alpha") = classify::dual_perceptron::default_alpha,
             py::arg("gamma") = classify::dual_perceptron::default_gamma,
             py::arg("bias") = classify::dual_perceptron::default_bias,
             py::arg("max_iter") = classify::dual_perceptron::default_max_iter)
        .def_readonly_static("default_alpha",
                             &classify::dual_perceptron::default_alpha)
        .def_readonly_static("default_gamma",
                             &classify::dual_perceptron::default_gamma)
        .def_readonly_static("default_bias",
                             &classify::dual_perceptron::default_bias)
        .def_readonly_static("default_max_iter",
                             &classify::dual_perceptron::default_max_iter);

    py::class_<classify::knn>{m_classify, "KNN", pycls}.def(
        "__init__",
        [](classify::knn& cls, classify::multiclass_dataset_view training,
           std::shared_ptr<index::inverted_index> idx, uint16_t k,
           const index::ranker& ranker, bool weighted) {
            std::stringstream ss;
            ranker.save(ss);

            new (&cls) classify::knn(std::move(training), std::move(idx), k,
                                     index::load_ranker(ss), weighted);
        },
        py::arg("training"), py::arg("inv_idx"), py::arg("k"),
        py::arg("ranker"), py::arg("weighted") = false);

    py::class_<classify::logistic_regression>{m_classify, "LogisticRegression",
                                              pycls}
        .def(py::init<classify::multiclass_dataset_view,
                      learn::sgd_model::options_type, double, uint64_t>(),
             py::arg("training"),
             py::arg("options") = learn::sgd_model::options_type{},
             py::arg("gamma") = classify::sgd::default_gamma,
             py::arg("max_iter") = classify::sgd::default_max_iter)
        .def("predict", &classify::logistic_regression::predict);

    py::class_<classify::naive_bayes>{m_classify, "NaiveBayes", pycls}
        .def(py::init<classify::multiclass_dataset_view, double, double>(),
             py::arg("training"),
             py::arg("alpha") = classify::naive_bayes::default_alpha,
             py::arg("beta") = classify::naive_bayes::default_beta)
        .def_readonly_static("default_alpha",
                             &classify::naive_bayes::default_alpha)
        .def_readonly_static("default_beta",
                             &classify::naive_bayes::default_beta);

    py::class_<classify::nearest_centroid>{m_classify, "NearestCentroid", pycls}
        .def(py::init<classify::multiclass_dataset_view,
                      std::shared_ptr<index::inverted_index>>(),
             py::arg("training"), py::arg("inv_idx"));

    py::class_<classify::one_vs_all>{m_classify, "OneVsAll", py_online_cls}.def(
        "__init__",
        [](classify::one_vs_all& ova, classify::multiclass_dataset_view mdv,
           py::object cls, py::kwargs kwargs) {

            auto creator = [=](const classify::binary_dataset_view& bdv) {
                // must acquire the GIL before calling back into Python
                // code to construct the classifier
                py::gil_scoped_acquire acq;
                return make_unique<cpp_created_py_binary_classifier>(
                    cls(bdv, **kwargs));
            };

            // release the GIL so that it can be re-acquired in the threads
            // that are spawned to create the sub-classifiers
            py::gil_scoped_release rel;
            new (&ova) classify::one_vs_all(std::move(mdv), std::move(creator));
        });

    py::class_<classify::one_vs_one>{m_classify, "OneVsOne", py_online_cls}.def(
        "__init__",
        [](classify::one_vs_one& ovo, classify::multiclass_dataset_view mdv,
           py::object cls, py::kwargs kwargs) {

            auto creator = [=](const classify::binary_dataset_view& bdv) {
                // must acquire the GIL before calling back into Python
                // code to construct the classifier
                py::gil_scoped_acquire acq;
                return make_unique<cpp_created_py_binary_classifier>(
                    cls(bdv, **kwargs));
            };

            // release the GIL so that it can be re-acquired in the threads
            // that are spawned to create the sub-classifiers
            py::gil_scoped_release rel;
            new (&ovo) classify::one_vs_one(std::move(mdv), std::move(creator));
        });

    py::class_<classify::winnow>{m_classify, "Winnow", pycls}
        .def(py::init<classify::multiclass_dataset_view, double, double,
                      std::size_t>(),
             py::arg("training"), py::arg("m") = classify::winnow::default_m,
             py::arg("gamma") = classify::winnow::default_gamma,
             py::arg("max_iter") = classify::winnow::default_max_iter)
        .def_readonly_static("default_m", &classify::winnow::default_m)
        .def_readonly_static("default_gamma", &classify::winnow::default_gamma)
        .def_readonly_static("default_max_iter",
                             &classify::winnow::default_max_iter);

    // utility functions
    m_classify.def(
        "cross_validate",
        [](std::function<py::object(classify::multiclass_dataset_view)> creator,
           classify::multiclass_dataset_view mdv, std::size_t k,
           bool even_split) {
            struct creator_type
            {
                py::object cls_;
                std::function<py::object(classify::multiclass_dataset_view)>&
                    creator_;

                creator_type(std::function<py::object(
                                 classify::multiclass_dataset_view)>& creator)
                    : creator_(creator)
                {
                    // nothing
                }

                classify::classifier*
                operator()(const classify::multiclass_dataset_view& mdv)
                {
                    cls_ = creator_(mdv);
                    return cls_.cast<classify::classifier*>();
                }
            } maker(creator);

            return classify::cross_validate(maker, mdv, k, even_split);
        },
        py::arg("creator"), py::arg("mdv"), py::arg("k"),
        py::arg("even_split") = false);
}
