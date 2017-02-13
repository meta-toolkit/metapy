/**
 * @file metapy_parser.cpp
 * @author Chase Geigle
 *
 * This file defines the metapy.parser submodule and creates bindings for
 * that part of the MeTA API.
 */

#include <cmath>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "meta/parser/trees/evalb.h"
#include "meta/parser/trees/internal_node.h"
#include "meta/parser/trees/leaf_node.h"
#include "meta/parser/trees/visitors/annotation_remover.h"
#include "meta/parser/trees/visitors/binarizer.h"
#include "meta/parser/trees/visitors/debinarizer.h"
#include "meta/parser/trees/visitors/empty_remover.h"
#include "meta/parser/trees/visitors/head_finder.h"
#include "meta/parser/trees/visitors/leaf_node_finder.h"
#include "meta/parser/trees/visitors/unary_chain_remover.h"

#include "meta/parser/sequence_extractor.h"
#include "meta/parser/sr_parser.h"

#include "meta/parser/io/ptb_reader.h"

#include "metapy_parser.h"

namespace py = pybind11;
using namespace meta;

template <class Visitor, class ResultType = typename Visitor::result_type>
class visitor_wrapper : public parser::visitor<py::object>
{
  public:
    virtual py::object operator()(parser::leaf_node& ln) override
    {
        return py::cast(vtor_(ln));
    }

    virtual py::object operator()(parser::internal_node& n) override
    {
        return py::cast(vtor_(n));
    }

    Visitor& visitor()
    {
        return vtor_;
    }

  private:
    Visitor vtor_;
};

template <class Visitor>
class visitor_wrapper<Visitor, std::unique_ptr<parser::node>>
    : public parser::visitor<py::object>
{
  public:
    virtual py::object operator()(parser::leaf_node& ln) override
    {
        return py::cast(vtor_(ln).release());
    }

    virtual py::object operator()(parser::internal_node& n) override
    {
        return py::cast(vtor_(n).release());
    }

    Visitor& visitor()
    {
        return vtor_;
    }

  private:
    Visitor vtor_;
};

template <class Visitor>
class visitor_wrapper<Visitor, void> : public parser::visitor<py::object>
{
  public:
    virtual py::object operator()(parser::leaf_node& ln) override
    {
        vtor_(ln);
        return py::cast(nullptr);
    }

    virtual py::object operator()(parser::internal_node& n) override
    {
        vtor_(n);
        return py::cast(nullptr);
    }

    Visitor& visitor()
    {
        return vtor_;
    }

  private:
    Visitor vtor_;
};

class py_visitor : public parser::visitor<py::object>
{
  public:
    virtual py::object operator()(parser::leaf_node& ln) override
    {
        PYBIND11_OVERLOAD_PURE(py::object, parser::visitor<py::object>,
                               visit_leaf, ln);
        return py::cast(nullptr);
    }

    virtual py::object operator()(parser::internal_node& n) override
    {

        PYBIND11_OVERLOAD_PURE(py::object, parser::visitor<py::object>,
                               visit_internal, n);
        return py::cast(nullptr);
    }
};

void metapy_bind_parser(py::module& m)
{
    using namespace parser;

    auto m_parse = m.def_submodule("parser");

    py::class_<node>{m_parse, "Node"}
        .def("category", &node::category)
        .def("is_leaf", &node::is_leaf)
        .def("is_temporary", &node::is_temporary)
        .def("equal", &node::equal)
        .def("accept", [](node& n, parser::visitor<py::object>& vtor) {
            return n.accept(vtor);
        });

    py::class_<leaf_node, node>{m_parse, "LeafNode"}
        .def(py::init<class_label, std::string>())
        .def("word", [](const leaf_node& ln) { return *ln.word(); });

    py::class_<internal_node, node>{m_parse, "InternalNode"}
        .def("__init__",
             [](internal_node& n, class_label cat, py::list pylist) {
                 std::vector<std::unique_ptr<node>> children(pylist.size());
                 for (std::size_t i = 0; i < pylist.size(); ++i)
                     children[i] = pylist[i].cast<node&>().clone();

                 new (&n) internal_node(std::move(cat), std::move(children));
             })
        .def(py::init<const internal_node&>())
        .def("add_child", [](internal_node& n,
                             const node& child) { n.add_child(child.clone()); })
        .def("num_children", &internal_node::num_children)
        .def("child", &internal_node::child, py::keep_alive<0, 1>())
        .def("head_lexicon", [](internal_node& n) { return n.head_lexicon(); },
             py::keep_alive<0, 1>())
        .def("head_lexicon",
             [](internal_node& n, const leaf_node* descendent) {
                 n.head_lexicon(descendent);
             })
        .def("head_constituent",
             [](internal_node& n) { return n.head_constituent(); },
             py::keep_alive<0, 1>())
        .def("head_constituent",
             [](internal_node& n, const node* descendent) {
                 n.head_constituent(descendent);
             })
        .def("each_child", [](internal_node& n, std::function<void(node*)> fn) {
            n.each_child(fn);
        });

    py::class_<parse_tree>{m_parse, "ParseTree"}
        .def("__init__",
             [](parse_tree& tree, const node& n) {
                 new (&tree) parse_tree(n.clone());
             })
        .def(py::init<const parse_tree&>())
        .def("__str__",
             [](const parse_tree& tree) {
                 std::stringstream ss;
                 ss << tree;
                 return ss.str();
             })
        .def("pretty_str",
             [](const parse_tree& tree) {
                 std::stringstream ss;
                 tree.pretty_print(ss);
                 return ss.str();
             })
        .def("visit", [](parse_tree& tree, parser::visitor<py::object>& vtor) {
            return tree.visit(vtor);
        });

    py::implicitly_convertible<node, parse_tree>();

    py::class_<visitor<py::object>, py_visitor> vtorbase{m_parse, "Visitor"};
    vtorbase.def(py::init<>())
        .def("visit_leaf",
             [](visitor<py::object>& vtor, leaf_node& ln) { return vtor(ln); })
        .def("visit_internal", [](visitor<py::object>& vtor,
                                  internal_node& in) { return vtor(in); });

    py::class_<visitor_wrapper<annotation_remover>>{
        m_parse, "AnnotationRemover", vtorbase}
        .def(py::init<>());
    py::class_<visitor_wrapper<binarizer>>{m_parse, "Binarizer", vtorbase}.def(
        py::init<>());
    py::class_<visitor_wrapper<debinarizer>>{m_parse, "Debinarizer", vtorbase}
        .def(py::init<>());
    py::class_<visitor_wrapper<empty_remover>>{m_parse, "EmptyRemover",
                                               vtorbase}
        .def(py::init<>());
    py::class_<visitor_wrapper<unary_chain_remover>>{
        m_parse, "UnaryChainRemover", vtorbase}
        .def(py::init<>());

    py::class_<visitor_wrapper<head_finder>>{m_parse, "HeadFinder", vtorbase}
        .def(py::init<>());
    py::class_<visitor_wrapper<leaf_node_finder>>{m_parse, "LeafNodeFinder",
                                                  vtorbase}
        .def(py::init<>())
        .def("leaves", [](visitor_wrapper<leaf_node_finder>& lnf) {
            // need to manually create the py::list here since the
            // pybind11 caster for vector operates on a const vector,
            // not a mutable one
            auto leaves = lnf.visitor().leaves();

            py::list ret(leaves.size());
            for (std::size_t i = 0; i < leaves.size(); ++i)
            {
                ret[i] = py::reinterpret_steal<py::object>(
                    py::detail::type_caster<std::unique_ptr<leaf_node>>::cast(
                        std::move(leaves[i]),
                        py::return_value_policy::automatic_reference,
                        py::handle()));
            }
            return ret;
        });

    py::class_<visitor_wrapper<sequence_extractor>>{
        m_parse, "SequenceExtractor", vtorbase}
        .def(py::init<>())
        .def("sequence", [](visitor_wrapper<sequence_extractor>& vtor) {
            return vtor.visitor().sequence();
        });

    py::class_<evalb>{m_parse, "EvalB"}
        .def(py::init<>())
        .def("matched", &evalb::matched)
        .def("proposed_total", &evalb::proposed_total)
        .def("gold_total", &evalb::gold_total)
        .def("labeled_precision", &evalb::labeled_precision)
        .def("labeled_recall", &evalb::labeled_recall)
        .def("labeled_f1", &evalb::labeled_f1)
        .def("perfect", &evalb::perfect)
        .def("average_crossing", &evalb::average_crossing)
        .def("zero_crossing", &evalb::zero_crossing)
        .def("add_tree", &evalb::add_tree);

    m_parse.def("extract_trees_from_file", [](const std::string& filename) {
        return parser::io::extract_trees(filename);
    });

    m_parse.def("extract_trees", [](const std::string& input) {
        std::stringstream ss{input};
        return parser::io::extract_trees(ss);
    });

    m_parse.def("read_tree", [](const std::string& input) {
        std::stringstream ss{input};
        return parser::io::extract_trees(ss).at(0);
    });

    py::class_<sr_parser> parser{m_parse, "Parser"};

    py::enum_<sr_parser::training_algorithm>{parser, "TrainingAlgorithm"}
        .value("EarlyTermination",
               sr_parser::training_algorithm::EARLY_TERMINATION)
        .value("BeamSearch", sr_parser::training_algorithm::BEAM_SEARCH);

    py::class_<sr_parser::training_options>{parser, "TrainingOptions"}
        .def(py::init<>())
        .def(py::init<const sr_parser::training_options&>())
        .def_readwrite("batch_size", &sr_parser::training_options::batch_size)
        .def_readwrite("beam_size", &sr_parser::training_options::beam_size)
        .def_readwrite("max_iterations",
                       &sr_parser::training_options::max_iterations)
        .def_readwrite("seed", &sr_parser::training_options::seed)
        .def_readwrite("num_threads", &sr_parser::training_options::num_threads)
        .def_readwrite("algorithm", &sr_parser::training_options::algorithm);

    parser.def(py::init<>())
        .def(py::init<const std::string&>())
        .def("parse", &sr_parser::parse)
        .def("train", &sr_parser::train)
        .def("save", &sr_parser::save);
}
