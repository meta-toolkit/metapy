/**
 * @file metapy_topics.h
 * @author Sean Massung
 */

#ifndef METAPY_TOPICS_H_
#define METAPY_TOPICS_H_

#include <cmath>
#include <pybind11/pybind11.h>

#include "meta/topics/topic_model.h"
#include "metapy_identifiers.h"

namespace pybind11
{
namespace detail
{
namespace metapy
{
template <class TupleType, class Identifier>
struct prob_caster
{
    PYBIND11_TYPE_CASTER(TupleType, _("Probability"));

    bool load(handle src, bool convert)
    {
        if (!isinstance<sequence>(src))
            return false;

        const auto seq = reinterpret_borrow<sequence>(src);
        if (seq.size() != 2)
            return false;

        if (!first.load(seq[0], convert) || !second.load(seq[1], convert))
            return false;
        value.tid = (Identifier)first;
        value.probability = (double)second;
        return true;
    }

    static handle cast(const TupleType& src, return_value_policy& policy,
                       handle& parent)
    {
        auto id = reinterpret_steal<object>(
            make_caster<meta::term_id>::cast(src.tid, policy, parent));
        auto prob = reinterpret_steal<object>(
            make_caster<double>::cast(src.probability, policy, parent));

        if (!id || !prob)
            return handle();

        tuple result(2);
        PyTuple_SET_ITEM(result.ptr(), 0, id.release().ptr());
        PyTuple_SET_ITEM(result.ptr(), 1, prob.release().ptr());
        return result.release();
    }

  protected:
    make_caster<Identifier> first;
    make_caster<double> second;
};
}

// add conversion for meta::topics::term_prob
template <>
struct type_caster<meta::topics::term_prob>
    : metapy::prob_caster<meta::topics::term_prob, meta::term_id>
{
};

// add conversion for meta::topics::topic_prob
template <>
struct type_caster<meta::topics::topic_prob>
    : metapy::prob_caster<meta::topics::topic_prob, meta::topic_id>
{
};
}
}

void metapy_bind_topics(pybind11::module& m);

#endif
