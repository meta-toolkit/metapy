/**
 * @file metapy_analyzers.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy.analyzers submodule and creates bindings for
 * that part of the MeTA API.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "metapy_analyzers.h"
#include "metapy_probe_map.h"

#include "cpptoml.h"
#include "meta/analyzers/token_stream.h"
#include "meta/analyzers/tokenizers/character_tokenizer.h"
#include "meta/analyzers/tokenizers/icu_tokenizer.h"
#include "meta/analyzers/filters/all.h"
#include "meta/analyzers/all.h"
#include "meta/corpus/document.h"

using namespace meta;

class py_token_stream
    : public util::clonable<analyzers::token_stream, py_token_stream>
{
  public:
    virtual std::string next() override
    {
        PYBIND11_OVERLOAD_PURE(std::string, analyzers::token_stream, next, );
        return "";
    }

    /**
     * Determines whether there are more tokens available in the
     * stream.
     */
    virtual operator bool() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, analyzers::token_stream, operator bool, );
        return false;
    }

    /**
     * Sets the content for the stream.
     * @param content The string content to set
     */
    virtual void set_content(std::string&& content) override
    {
        PYBIND11_OVERLOAD_PURE(void, analyzers::token_stream, set_content,
                               std::move(content));
    }
};

class py_analyzer : public util::clonable<analyzers::analyzer, py_analyzer>
{
    virtual void tokenize(const corpus::document& doc,
                          analyzers::featurizer& counts) override
    {
        PYBIND11_OVERLOAD_PURE(void, analyzers::analyzer, tokenize, doc,
                               counts);
    }
};

template <class TokenStream, class... Args>
void make_token_stream(TokenStream& next, const analyzers::token_stream& prev,
                       Args... args)
{
    new (&next) TokenStream(prev.clone(), args...);
}

namespace py = pybind11;

void metapy_bind_analyzers(py::module& m)
{
    using namespace analyzers;

    auto m_ana = m.def_submodule("analyzers");

    py::class_<py_token_stream> ts_base{m_ana, "TokenStream"};
    ts_base.alias<token_stream>()
        .def(py::init<>())
        .def("next",
             [](token_stream& ts)
             {
                 if (!ts)
                     throw py::stop_iteration();
                 return ts.next();
             })
        .def("has_more",
             [](const token_stream& ts)
             {
                 return static_cast<bool>(ts);
             })
        .def("set_content",
             [](token_stream& ts, std::string str)
             {
                 ts.set_content(std::move(str));
             })
        .def("__iter__",
             [](token_stream& ts) -> token_stream&
             {
                 return ts;
             })
        .def("__next__", [](token_stream& ts)
             {
                 if (!ts)
                     throw py::stop_iteration();
                 return ts.next();
             });

    // tokenizers
    py::class_<tokenizers::character_tokenizer>{m_ana, "CharacterTokenizer",
                                                ts_base}
        .def(py::init<>());

    py::class_<tokenizers::icu_tokenizer>{m_ana, "ICUTokenizer", ts_base}.def(
        py::init<bool>(),
        "Creates a tokenizer using the UTF text segmentation standard",
        py::arg("suppress_tags") = false);

    // filters
    py::class_<filters::alpha_filter>{m_ana, "AlphaFilter", ts_base}.def(
        "__init__", &make_token_stream<filters::alpha_filter>);

    py::class_<filters::empty_sentence_filter>{m_ana, "EmptySentenceFilter",
                                               ts_base}
        .def("__init__", &make_token_stream<filters::empty_sentence_filter>);

    py::class_<filters::english_normalizer>{m_ana, "EnglishNormalizer", ts_base}
        .def("__init__", &make_token_stream<filters::english_normalizer>);

    py::class_<filters::icu_filter>{m_ana, "ICUFilter", ts_base}.def(
        "__init__",
        &make_token_stream<filters::icu_filter, const std::string&>);

    py::class_<filters::length_filter>{m_ana, "LengthFilter", ts_base}.def(
        "__init__",
        &make_token_stream<filters::length_filter, uint64_t, uint64_t>,
        py::arg("source"), py::arg("min"), py::arg("max"));

    py::class_<filters::list_filter> list_filter{m_ana, "ListFilter", ts_base};
    py::enum_<filters::list_filter::type>{list_filter, "Type"}
        .value("Accept", filters::list_filter::type::ACCEPT)
        .value("Reject", filters::list_filter::type::REJECT);
    list_filter.def("__init__",
                    &make_token_stream<filters::list_filter, const std::string&,
                                       filters::list_filter::type>);

    py::class_<filters::lowercase_filter>{m_ana, "LowercaseFilter", ts_base}
        .def("__init__", &make_token_stream<filters::lowercase_filter>);

    py::class_<filters::porter2_filter>{m_ana, "Porter2Filter", ts_base}.def(
        "__init__", &make_token_stream<filters::porter2_filter>);

    py::class_<filters::ptb_normalizer>{m_ana, "PennTreebankNormalizer",
                                        ts_base}
        .def("__init__", &make_token_stream<filters::ptb_normalizer>);

    py::class_<filters::sentence_boundary>{m_ana, "SentenceBoundaryAdder",
                                           ts_base}
        .def("__init__", &make_token_stream<filters::sentence_boundary>);


    // analyzers
    py::class_<py_analyzer> analyzer_base{m_ana, "Analyzer"};
    analyzer_base.alias<analyzers::analyzer>()
        .def(py::init<>())
        .def("analyze", &analyzer::analyze<uint64_t>)
        .def("featurize", &analyzer::analyze<double>);

    py::class_<ngram_word_analyzer>{m_ana, "NGramWordAnalyzer", analyzer_base}
        .def("__init__",
             [](ngram_word_analyzer& ana, uint16_t n, const token_stream& ts)
             {
                 new (&ana) ngram_word_analyzer(n, ts.clone());
             });

    py::class_<multi_analyzer>{m_ana, "MultiAnalyzer", analyzer_base};

    m_ana.def("load", [](const std::string& filename)
              {
                  auto config = cpptoml::parse_file(filename);
                  return analyzers::load(*config);
              });
}
