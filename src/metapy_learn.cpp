/**
 * @file metapy_learn.cpp
 * @author Chase Geigle
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpptoml.h"
#include "meta/learn/dataset.h"
#include "meta/learn/dataset_view.h"
#include "meta/learn/loss/all.h"
#include "meta/learn/sgd.h"
#include "meta/learn/transform.h"
#include "meta/util/iterator.h"
#include "metapy_identifiers.h"
#include "metapy_learn.h"

namespace py = pybind11;
using namespace meta;

template <class LossFunction, class Base>
void bind_loss_function(py::module& m, const char* name, Base& base)
{
    py::class_<LossFunction>{m, name, base}
        .def(py::init<>())
        .def_property_readonly_static("id", [](py::object /* self */) {
            return LossFunction::id.to_string();
        });
}

struct py_loss_function : public learn::loss::loss_function
{
    double loss(double prediction, double expected) const override
    {
        PYBIND11_OVERLOAD_PURE(double, learn::loss::loss_function, loss,
                               prediction, expected);
        return 0;
    }

    double derivative(double prediction, double expected) const override
    {
        PYBIND11_OVERLOAD_PURE(double, learn::loss::loss_function, derivative,
                               prediction, expected);
        return 0;
    }

    virtual void save(std::ostream& /* os */) const override
    {
        throw std::runtime_error{
            "cannot serialize python-defined loss functions"};
    }
};

void metapy_bind_learn(py::module& m)
{
    auto m_learn = m.def_submodule("learn");

    py::class_<learn::feature_vector>{m_learn, "FeatureVector"}
        .def(py::init<>())
        .def(py::init<uint64_t>())
        .def(py::init<const learn::feature_vector&>())
        .def("__init__",
             [](learn::feature_vector& fv, py::iterable& iter) {
                 using pair_type = learn::feature_vector::pair_type;
                 auto cast_fn
                     = [](py::handle h) { return h.cast<pair_type>(); };
                 new (&fv) learn::feature_vector(
                     util::make_transform_iterator(iter.begin(), cast_fn),
                     util::make_transform_iterator(iter.end(), cast_fn));
             })
        .def("__len__", &learn::feature_vector::size)
        .def("__iter__",
             [](learn::feature_vector& fv) {
                 return py::make_iterator(fv.begin(), fv.end());
             },
             py::keep_alive<0, 1>())
        .def("__getitem__", [](const learn::feature_vector& fv,
                               learn::feature_id fid) { return fv.at(fid); })
        .def("__setitem__", [](learn::feature_vector& fv, learn::feature_id fid,
                               double val) { fv[fid] = val; })
        .def("clear", &learn::feature_vector::clear)
        .def("shrink_to_fit", &learn::feature_vector::shrink_to_fit)
        .def("condense", &learn::feature_vector::condense)
        .def("dot",
             [](const learn::feature_vector& self,
                const learn::feature_vector& other) {
                 return util::dot_product(self, other);
             })
        .def("cosine",
             [](const learn::feature_vector& self,
                const learn::feature_vector& other) {
                 return util::cosine_sim(self, other);
             })
        .def("l2norm",
             [](const learn::feature_vector& self) {
                 return util::l2norm(self);
             })
        .def("__str__", [](const learn::feature_vector& fv) {
            std::stringstream ss;
            util::string_view padding = "";
            ss << '[';
            for (const auto& pr : fv)
            {
                ss << padding << '(' << pr.first << ", " << pr.second << ')';
                padding = ", ";
            }
            ss << ']';
            return ss.str();
        });

    m_learn.def("dot", &util::dot_product<const learn::feature_vector&,
                                          const learn::feature_vector&>);
    m_learn.def("cosine", &util::cosine_sim<const learn::feature_vector&,
                                            const learn::feature_vector&>);
    m_learn.def("l2norm", [](const learn::feature_vector& vec) {
        return util::l2norm(vec);
    });

    py::class_<learn::instance>{m_learn, "Instance"}
        .def(py::init<learn::instance_id>())
        .def(py::init<learn::instance_id, learn::feature_vector>())
        .def_readonly("id", &learn::instance::id)
        .def_readwrite("weights", &learn::instance::weights);

    py::class_<learn::dataset> pydset{m_learn, "Dataset"};
    pydset
        .def("__init__",
             [](learn::dataset& dset,
                const std::shared_ptr<index::forward_index>& fidx) {
                 py::gil_scoped_release release;
                 new (&dset) learn::dataset(fidx);
             })
        .def("__init__",
             [](learn::dataset& dset,
                const std::shared_ptr<index::forward_index>& fidx,
                const std::vector<doc_id>& docs) {
                 py::gil_scoped_release release;
                 new (&dset) learn::dataset(fidx, docs);
             })
        .def("__init__",
             [](learn::dataset& dset, py::list& data,
                std::size_t total_features, py::function& featurizer) {
                 new (&dset)
                     learn::dataset(data.begin(), data.end(), total_features,
                                    [&](py::handle obj) {
                                        return py::cast<learn::feature_vector>(
                                            featurizer(obj));
                                    });
             })
        .def("__getitem__",
             [](learn::dataset& dset, int64_t offset) -> learn::instance& {
                 std::size_t idx = offset >= 0
                                       ? static_cast<std::size_t>(offset)
                                       : dset.size() + offset;
                 if (idx >= dset.size())
                     throw py::index_error();
                 return *(dset.begin() + idx);
             },
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             [](learn::dataset& dset, py::slice slice) {
                 learn::dataset_view dv{dset};
                 return make_sliced_dataset_view(dv, slice);
             },
             py::keep_alive<0, 1>())
        .def("__len__", &learn::dataset::size)
        .def("__iter__",
             [](const learn::dataset& dset) {
                 return py::make_iterator(dset.begin(), dset.end());
             },
             py::keep_alive<0, 1>())
        .def("total_features", &learn::dataset::total_features);

    py::class_<learn::dataset_view>{m_learn, "DatasetView"}
        .def(py::init<const learn::dataset&>(), py::keep_alive<0, 1>())
        .def("shuffle", &learn::dataset_view::shuffle)
        .def("rotate", &learn::dataset_view::rotate)
        .def("total_features", &learn::dataset_view::total_features)
        .def("__getitem__",
             [](const learn::dataset_view& dv, int64_t offset) {
                 std::size_t idx = offset >= 0
                                       ? static_cast<std::size_t>(offset)
                                       : dv.size() + offset;
                 if (idx >= dv.size())
                     throw py::index_error();
                 return *(dv.begin() + idx);
             })
        .def("__getitem__",
             [](const learn::dataset_view& dv, py::slice slice) {
                 return make_sliced_dataset_view(dv, slice);
             },
             py::keep_alive<0, 1>())
        .def("__len__", &learn::dataset_view::size)
        .def("__iter__",
             [](const learn::dataset_view& dv) {
                 return py::make_iterator(dv.begin(), dv.end());
             },
             py::keep_alive<0, 1>());

    py::implicitly_convertible<learn::dataset, learn::dataset_view>();

    m_learn.def("tfidf_transform", &learn::tfidf_transform);
    m_learn.def("l2norm_transform", &learn::l2norm_transform);

    auto m_loss = m_learn.def_submodule("loss");

    py::class_<learn::loss::loss_function, py_loss_function> pyloss{
        m_loss, "LossFunction"};
    pyloss.def("loss", &learn::loss::loss_function::loss)
        .def("derivative", &learn::loss::loss_function::derivative);

    bind_loss_function<learn::loss::hinge>(m_loss, "Hinge", pyloss);
    bind_loss_function<learn::loss::huber>(m_loss, "Huber", pyloss);
    bind_loss_function<learn::loss::least_squares>(m_loss, "LeastSquares",
                                                   pyloss);
    bind_loss_function<learn::loss::logistic>(m_loss, "Logistic", pyloss);
    bind_loss_function<learn::loss::modified_huber>(m_loss, "ModifiedHuber",
                                                    pyloss);
    bind_loss_function<learn::loss::perceptron>(m_loss, "Perceptron", pyloss);
    bind_loss_function<learn::loss::smooth_hinge>(m_loss, "SmoothHinge",
                                                  pyloss);
    bind_loss_function<learn::loss::squared_hinge>(m_loss, "SquaredHinge",
                                                   pyloss);

    py::class_<learn::sgd_model> py_sgdmodel{m_learn, "SGDModel"};
    py::class_<learn::sgd_model::options_type>{py_sgdmodel, "Options"}
        .def(py::init<>())
        .def_readwrite("learning_rate",
                       &learn::sgd_model::options_type::learning_rate)
        .def_readwrite("l2_regularizer",
                       &learn::sgd_model::options_type::l2_regularizer)
        .def_readwrite("l1_regularizer",
                       &learn::sgd_model::options_type::l1_regularizer);
    py_sgdmodel
        .def_readonly_static("default_learning_rate",
                             &learn::sgd_model::default_learning_rate)
        .def_readonly_static("default_l2_regularizer",
                             &learn::sgd_model::default_l2_regularizer)
        .def_readonly_static("default_l1_regularizer",
                             &learn::sgd_model::default_l1_regularizer)
        .def(py::init<std::size_t, learn::sgd_model::options_type>())
        .def("predict", &learn::sgd_model::predict)
        .def("train_one", &learn::sgd_model::train_one);
}
