/**
 * @file metapy.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy module and bindings for the MeTA API. It
 * does not attempt to be completely comprehensive at this time (though
 * that is an eventual goal), but it aims to provide at least enough of an
 * API surface so that interactive web demos can be made.
 */

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "cpptoml.h"
#include "meta/index/inverted_index.h"
#include "meta/index/make_index.h"
#include "meta/index/score_data.h"
#include "meta/index/ranker/all.h"

namespace pybind11
{
namespace detail
{

// add conversion for hashing::probe_map
// @see pybind11/stl.h
template <class Type, class Key, class Value>
struct probe_map_caster
{
    using type = Type;
    using key_conv = type_caster<typename intrinsic_type<Key>::type>;
    using value_conv = type_caster<typename intrinsic_type<Value>::type>;

    bool load(handle src, bool convert)
    {
        dict d{src, true};
        if (!d.check())
            return false;
        key_conv kconv;
        value_conv vconv;
        value.clear();
        for (auto it : d)
        {
            if (!kconv.load(it.first.ptr(), convert)
                || !vconv.load(it.second.ptr(), convert))
                return false;
            value.emplace((Key)kconv, (Value)vconv);
        }
        return true;
    }

    static handle cast(const type& src, return_value_policy policy,
                       handle parent)
    {
        dict d;
        for (const auto& kv : src)
        {
            object key{key_conv::cast(kv.key(), policy, parent), false};
            object value{value_conv::cast(kv.value(), policy, parent), false};
            if (!key || !value)
                return handle{};
            d[key] = value;
        }
        return d.release();
    }

    PYBIND11_TYPE_CASTER(type, _("dict<") + key_conv::name() + _(", ")
                                   + value_conv::name() + _(">"))
};

template <class Key, class Value, class ProbingStrategy, class Hash,
          class KeyEqual, class Traits>
struct type_caster<meta::hashing::probe_map<Key, Value, ProbingStrategy, Hash,
                                            KeyEqual, Traits>>
    : probe_map_caster<meta::hashing::probe_map<Key, Value, ProbingStrategy,
                                                Hash, KeyEqual, Traits>,
                       Key, Value>
{
};

// add conversion for meta::index::search_result
// see: pybind11/cast.h
template <>
struct type_caster<meta::index::search_result>
    : type_caster<std::pair<meta::doc_id, float>>
{
    using type = meta::index::search_result;
    using base = type_caster<std::pair<meta::doc_id, float>>;

    using base::load;
    using base::name;

    static handle cast(const type& src, return_value_policy policy,
                       handle parent)
    {
        object o1 = object(
            type_caster<typename intrinsic_type<meta::doc_id>::type>::cast(
                src.d_id, policy, parent),
            false);
        object o2
            = object(type_caster<typename intrinsic_type<float>::type>::cast(
                         src.score, policy, parent),
                     false);
        if (!o1 || !o2)
            return handle();

        tuple result(2);
        PyTuple_SET_ITEM(result.ptr(), 0, o1.release().ptr());
        PyTuple_SET_ITEM(result.ptr(), 1, o2.release().ptr());
        return result.release();
    }

    operator type()
    {
        auto pr = static_cast<std::pair<meta::doc_id, float>>(*this);

        return type{pr.first, pr.second};
    }
};
}
}

namespace py = pybind11;

using namespace meta;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

class py_ranker : public index::ranker
{
  public:
    using index::ranker::ranker;

    float score_one(const meta::index::score_data& sd) override
    {
        PYBIND11_OVERLOAD_PURE(float, index::ranker, score_one, sd);
        return 0.0f;
    }

    void save(std::ostream&) const override
    {
        throw std::runtime_error{"cannot serialize python-defined rankers"};
    }
};

class py_lm_ranker : public index::language_model_ranker
{
  public:
    using index::language_model_ranker::language_model_ranker;

    float smoothed_prob(const index::score_data& sd) const override
    {
        PYBIND11_OVERLOAD_PURE(float, index::language_model_ranker,
                               smoothed_prob, sd);
        return 0.0f;
    }

    float doc_constant(const index::score_data& sd) const override
    {
        PYBIND11_OVERLOAD_PURE(float, index::language_model_ranker,
                               doc_constant, sd);
        return 0.0f;
    }

    void save(std::ostream&) const override
    {
        throw std::runtime_error{"cannot serialize python-defined rankers"};
    }
};

PYBIND11_PLUGIN(metapy)
{
    py::module m{"metapy", "MeTA toolkit python bindings"};

    py::module m_idx = m.def_submodule("index");

    py::class_<corpus::document>{m_idx, "Document"}
        .def(py::init<doc_id, const class_label&>(),
             py::arg("d_id") = doc_id{0},
             py::arg("label") = class_label{"[NONE]"})
        .def("label",
             [](const corpus::document& doc)
             {
                 return doc.label();
             },
             "Gets the label for the document")
        .def("label",
             [](corpus::document& doc, const class_label& label)
             {
                 doc.label(label);
             },
             "Sets the label for the document")
        .def("content",
             [](const corpus::document& doc)
             {
                 return doc.content();
             },
             "Gets the content of the document")
        .def("content",
             [](corpus::document& doc, const std::string& content,
                const std::string& encoding)
             {
                 doc.content(content, encoding);
             },
             "Sets the content of the document", py::arg("content"),
             py::arg("encoding") = "utf-8")
        .def("encoding",
             [](const corpus::document& doc)
             {
                 return doc.encoding();
             },
             "Gets the encoding for the document's content")
        .def("encoding",
             [](corpus::document& doc, const std::string& encoding)
             {
                 doc.encoding(encoding);
             },
             "Sets the encoding for the document's content")
        .def("id", &corpus::document::id)
        .def("contains_content", &corpus::document::contains_content);

    py::class_<index::disk_index>{m_idx, "DiskIndex"}
        .def("index_name", &index::disk_index::index_name)
        .def("num_docs", &index::disk_index::num_docs)
        .def("doc_name", &index::disk_index::doc_name)
        .def("doc_path", &index::disk_index::doc_path)
        .def("docs", &index::disk_index::docs)
        .def("doc_size", &index::disk_index::doc_size)
        .def("label", &index::disk_index::label)
        .def("lbl_id", &index::disk_index::lbl_id)
        .def("class_label_from_id", &index::disk_index::class_label_from_id)
        .def("num_labels", &index::disk_index::num_labels)
        .def("class_labels", &index::disk_index::class_labels)
        .def("unique_terms",
             [](const index::disk_index& idx)
             {
                 return idx.unique_terms();
             })
        .def("unique_terms",
             [](const index::disk_index& idx, doc_id did)
             {
                 return idx.unique_terms(did);
             })
        .def("get_term_id", &index::disk_index::get_term_id)
        .def("term_text", &index::disk_index::term_text);

    py::class_<index::inverted_index, std::shared_ptr<index::inverted_index>>{
        m_idx, "InvertedIndex", py::base<index::disk_index>{}}
        .def("tokenize", &index::inverted_index::tokenize)
        .def("doc_freq", &index::inverted_index::doc_freq)
        .def("term_freq", &index::inverted_index::term_freq)
        .def("total_corpus_terms", &index::inverted_index::total_corpus_terms)
        .def("total_num_occurences",
             &index::inverted_index::total_num_occurences)
        .def("avg_doc_length", &index::inverted_index::avg_doc_length);

    m_idx.def("make_inverted_index",
              [](const std::string& filename)
              {
                  auto config = cpptoml::parse_file(filename);
                  return index::make_index<index::inverted_index>(*config);
              },
              "Builds or loads an inverted index from disk");

    py::class_<index::score_data>{m_idx, "ScoreData"}
        .def(py::init<index::inverted_index&, float, uint64_t, uint64_t,
                      float>())
        .def_property_readonly(
            "idx",
            [](index::score_data& sd) -> index::inverted_index&
            {
                return sd.idx;
            })
        .def_readwrite("avg_dl", &index::score_data::avg_dl)
        .def_readwrite("num_docs", &index::score_data::num_docs)
        .def_readwrite("total_terms", &index::score_data::total_terms)
        .def_readwrite("query_length", &index::score_data::query_length)
        .def_readwrite("t_id", &index::score_data::t_id)
        .def_readwrite("query_term_weight",
                       &index::score_data::query_term_weight)
        .def_readwrite("doc_count", &index::score_data::doc_count)
        .def_readwrite("corpus_term_count",
                       &index::score_data::corpus_term_count)
        .def_readwrite("d_id", &index::score_data::d_id)
        .def_readwrite("doc_term_count", &index::score_data::doc_term_count)
        .def_readwrite("doc_size", &index::score_data::doc_size)
        .def_readwrite("doc_unique_terms",
                       &index::score_data::doc_unique_terms);

    py::class_<py_ranker> rank_base{m_idx, "Ranker"};
    rank_base.alias<index::ranker>()
        .def(py::init<>())
        .def("score",
             [](index::ranker& ranker, index::inverted_index& idx,
                const corpus::document& query, uint64_t num_results,
                const index::ranker::filter_function_type& filter)
             {
                 return ranker.score(idx, query, num_results, filter);
             },
             "Scores the documents in the inverted index with respect to the "
             "query "
             "using this ranker",
             py::arg("idx"), py::arg("query"), py::arg("num_results") = 10,
             py::arg("filter") = std::function<bool(doc_id)>([](doc_id)
                                                             {
                                                                 return true;
                                                             }))
        .def("score_one", &index::ranker::score_one);

    py::class_<py_lm_ranker> lm_rank_base{m_idx, "LanguageModelRanker",
                                          rank_base};
    lm_rank_base.alias<index::language_model_ranker>().def(py::init<>());

    py::class_<index::absolute_discount>{m_idx, "AbsoluteDiscount",
                                         lm_rank_base}
        .def(py::init<float>(),
             py::arg("delta") = index::absolute_discount::default_delta);

    py::class_<index::dirichlet_prior>{m_idx, "DirichletPrior", lm_rank_base}
        .def(py::init<float>(),
             py::arg("mu") = index::dirichlet_prior::default_mu);

    py::class_<index::jelinek_mercer>{m_idx, "JelinekMercer", lm_rank_base}.def(
        py::init<float>(),
        py::arg("lambda") = index::jelinek_mercer::default_lambda);

    py::class_<index::pivoted_length>{m_idx, "PivotedLength", rank_base}.def(
        py::init<float>(), py::arg("s") = index::pivoted_length::default_s);

    py::class_<index::okapi_bm25>{m_idx, "OkapiBM25", rank_base}.def(
        py::init<float, float, float>(),
        py::arg("k1") = index::okapi_bm25::default_k1,
        py::arg("b") = index::okapi_bm25::default_b,
        py::arg("k3") = index::okapi_bm25::default_k3);

    return m.ptr();
}
