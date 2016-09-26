/**
 * @file metapy_analyzers.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy.analyzers submodule and creates bindings for
 * that part of the MeTA API.
 */

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <mutex>

#include "metapy_analyzers.h"
#include "metapy_probe_map.h"

#include "cpptoml.h"
#include "meta/analyzers/all.h"
#include "meta/analyzers/filters/all.h"
#include "meta/analyzers/token_stream.h"
#include "meta/analyzers/tokenizers/character_tokenizer.h"
#include "meta/analyzers/tokenizers/icu_tokenizer.h"
#include "meta/corpus/document.h"
#include "meta/parallel/thread_pool.h"
#include "meta/util/algorithm.h"

namespace py = pybind11;
using namespace meta;

/**
 * This class is a "trampoline" class to bounce functions back to Python
 * if they are overloaded there rather than in C++ directly.
 */
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
#if PY_MAJOR_VERSION < 3
        PYBIND11_OVERLOAD_PURE_NAME(bool, analyzers::token_stream,
                                    "__nonzero__", operator bool,);
#else
        PYBIND11_OVERLOAD_PURE_NAME(bool, analyzers::token_stream,
                                    "__bool__", operator bool,);
#endif
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

/**
 * This class holds a token_stream that was defined in Python, but was
 * created using C++.
 *
 * This is where stuff gets weird. We want our tokenizer_factory to return
 * std::unique_ptr<token_stream> when invoked with an id and a config
 * group. The problem is that we don't have a good way to get a unique_ptr
 * out of Python code if the token stream is defined there, since that
 * basically entails having Python relinquish ownership of something, which
 * it isn't apt to do.
 *
 * Instead, what we do is have a separate class that can wrap an object
 * created by invoking Python code directly from C++. This doesn't give us
 * a unique_ptr, but we can enforce that ourselves directly. The object
 * will down convert nicely to a token_stream through pybind11's casting
 * utilities, since it is still a token_stream derivative at heart. We just
 * can't get at its unique_ptr. We store a py::object to keep the reference
 * count > 0, and then a token_stream* that we do all of the actual work
 * with. We have to override all of the virtual functions again, but that
 * isn't too much work.
 *
 * Since we're not using the PYBIND11_OVERLOAD functions anymore (since
 * this object isn't the one registered with Python), we have to be careful
 * with the GIL. Each function here acquires the GIL immediately before
 * doing anything else so that hitting Python is safely behind the lock.
 * This means things are going to be a lot slower, of course, but it's the
 * only way I can think of for doing this safely for now.
 *
 * We also want to be able to clone token_streams, since that's how we set
 * up the pipeline replicas across all of the threads. We can do that by
 * providing a copy constructor that calls into Python by invoking
 * `copy.deepcopy(obj)` to copy our current Python object.
 *
 * Finally, our destructor is weird since we want to decrement the object's
 * reference count while still inside the GIL.
 */
class cpp_created_py_token_stream
    : public util::clonable<analyzers::token_stream,
                            cpp_created_py_token_stream>
{
  public:
    cpp_created_py_token_stream(py::object obj)
        : obj_{obj}, stream_{obj_.cast<token_stream*>()}
    {
        // nothing
    }

    cpp_created_py_token_stream(const cpp_created_py_token_stream& other)
    {
        py::gil_scoped_acquire acq;
        auto deepcopy = py::module::import("copy").attr("deepcopy");
        obj_ = deepcopy.cast<py::function>()(other.obj_);
        stream_ = obj_.cast<token_stream*>();
    }

    virtual std::string next() override
    {
        py::gil_scoped_acquire acq;
        return stream_->next();
    }

    virtual operator bool() const override
    {
        py::gil_scoped_acquire acq;
        return *stream_;
    }

    virtual void set_content(std::string&& content) override
    {
        py::gil_scoped_acquire acq;
        stream_->set_content(std::move(content));
    }

    ~cpp_created_py_token_stream()
    {
        py::gil_scoped_acquire acq;
        obj_.release().dec_ref();
    }

  private:
    py::object obj_;
    token_stream* stream_;
};

/**
 * Registers a Python object with a factory.
 *
 * There are two major assumptions here. First, we assume that the *Class*
 * object passed here has an "id" property, just like we assume the MeTA
 * classes do in C++.
 *
 * We have to do a bit of trickery here, though. The factories typically
 * map util::string_view to creation functions. This is fine in the C++
 * code, since every id ends up being a static C string somewhere in the
 * data segment. But here, the ids are in Python and are dynamically
 * allocated. To save ourselves headache and prevent UB, we have a static
 * function level cache here to store the strings we've added to the
 * factories so that the util::string_view will be valid.
 *
 * There's probably a better way of doing this, but this currently works.
 */
template <class FactoryType, class CreationFunction>
void py_factory_register(py::object cls, FactoryType& factory,
                         CreationFunction&& c_fun)
{
    static std::vector<std::string> ids;
    static std::mutex mut;
    util::string_view id;
    {
        std::lock_guard<std::mutex> lock{mut};
        ids.push_back(cls.attr("id").cast<std::string>());
        id = ids.back();
    }
    std::cerr << "filter_factory adding " << id << std::endl;
    factory.add(id, c_fun);
}

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

template <class T>
py::object ngram_analyze(analyzers::ngram_word_analyzer& ana,
                         const corpus::document& doc)
{
    if (ana.n_value() == 1)
        return py::cast(ana.analyze<T>(doc));

    auto ngrams = ana.analyze<T>(doc);

    py::dict ret;
    for (const auto& kv : ngrams)
    {
        const auto& key = kv.key();

        using iterator = decltype(key.begin());

        py::tuple newkey{ana.n_value()};
        uint64_t idx = 0;
        util::for_each_token(key.begin(), key.end(), "_",
                             [&](iterator first, iterator last) {
                                 if (first != last)
                                     newkey[idx++] = py::str({first, last});
                             });
        ret[newkey] = py::cast(kv.value());
    }

    return ret;
}

/**
 * A visitor class for converting a TOML configuration group to a Python
 * dictionary. We use this to convert TOML tables to keyword arguments for
 * token_streams defined in Python.
 */
class py_toml_visitor
{
  public:
    template <class T>
    void visit(const cpptoml::value<T>& v, py::object& obj)
    {
        obj = py::cast(v.get());
    }

    void visit(const cpptoml::table& table, py::object& obj)
    {
        obj = py::dict();
        auto dict = obj.cast<py::dict>();

        for (const auto& pr : table)
        {
            auto key = py::cast(pr.first);
            py::object value;
            pr.second->accept(*this, value);
            dict[key] = value;
        }
    }

    void visit(const cpptoml::array& arr, py::object& obj)
    {
        obj = py::list();
        auto lst = obj.cast<py::list>();
        for (const auto& val : arr)
        {
            py::object value;
            val->accept(*this, value);
            lst.append(value);
        }
    }

    void visit(const cpptoml::table_array& tarr, py::object& obj)
    {
        obj = py::list();
        auto lst = obj.cast<py::list>();
        for (const auto& table : tarr)
        {
            py::object value;
            table->accept(*this, value);
            lst.append(value);
        }
    }
};

class py_token_stream_iterator
{
    analyzers::token_stream& stream_;
    py::object ref_;

  public:
    py_token_stream_iterator(analyzers::token_stream& stream, py::object ref)
        : stream_(stream), ref_(ref)
    {
        // nothing
    }

    std::string next()
    {
        if (!stream_)
            throw py::stop_iteration();
        return stream_.next();
    }
};

void metapy_bind_analyzers(py::module& m)
{
    using namespace analyzers;

    auto m_ana = m.def_submodule("analyzers");

    py::class_<token_stream, py_token_stream> ts_base{m_ana, "TokenStream"};
    ts_base.def(py::init<>())
        .def("next",
             [](token_stream& ts) {
                 if (!ts)
                     throw py::stop_iteration();
                 return ts.next();
             })
        .def("set_content",
             [](token_stream& ts, std::string str) {
                 ts.set_content(std::move(str));
             })
        .def("__bool__", [](token_stream& ts) { return static_cast<bool>(ts); })
        .def("__iter__",
             [](py::object ts) {
                 return py_token_stream_iterator(ts.cast<token_stream&>(), ts);
             })
        .def("__deepcopy__",
             [](token_stream& ts, py::dict&) { return ts.clone(); });

    py::class_<py_token_stream_iterator>(ts_base, "Iterator")
        .def("__iter__",
             [](py_token_stream_iterator& it) -> py_token_stream_iterator& {
                 return it;
             })
        .def("__next__", &py_token_stream_iterator::next);

    // tokenizers
    py::class_<tokenizers::character_tokenizer>{m_ana, "CharacterTokenizer",
                                                ts_base}
        .def(py::init<>());

    py::class_<tokenizers::icu_tokenizer>{m_ana, "ICUTokenizer", ts_base}.def(
        py::init<bool>(),
        "Creates a tokenizer using the UTF text segmentation standard",
        // hack around g++ 4.8 ambiguous overloaded operator=
        py::arg_t<bool>{"suppress_tags", false});
    // py::arg("suppress_tags") = false);

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
    py::class_<analyzers::analyzer, py_analyzer> analyzer_base{m_ana,
                                                               "Analyzer"};
    analyzer_base.def(py::init<>())
        .def("analyze", &analyzer::analyze<uint64_t>)
        .def("featurize", &analyzer::analyze<double>);

    py::class_<ngram_word_analyzer>{m_ana, "NGramWordAnalyzer", analyzer_base}
        .def("__init__",
             [](ngram_word_analyzer& ana, uint16_t n, const token_stream& ts) {
                 new (&ana) ngram_word_analyzer(n, ts.clone());
             })
        .def("analyze", &ngram_analyze<uint64_t>)
        .def("featurize", &ngram_analyze<double>);

    py::class_<multi_analyzer>{m_ana, "MultiAnalyzer", analyzer_base};

    m_ana.def("load", [](const std::string& filename) {
        auto config = cpptoml::parse_file(filename);
        return analyzers::load(*config);
    });

    m_ana.def("register_filter", [](py::object cls) {
        py_factory_register(cls, filter_factory::get(),
                            [=](std::unique_ptr<token_stream> source,
                                const cpptoml::table& cfg) {
                                py::gil_scoped_acquire acq;

                                py::dict kwargs;
                                py_toml_visitor vtor;
                                cfg.accept(vtor, kwargs);
                                PyDict_DelItemString(kwargs.ptr(), "type");

                                return make_unique<cpp_created_py_token_stream>(
                                    cls(source->clone(), **kwargs));
                            });
    });
}
