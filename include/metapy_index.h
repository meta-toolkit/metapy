/**
 * @file metapy_index.h
 * @author Chase Geigle
 *
 */

#ifndef METAPY_INDEX_H_
#define METAPY_INDEX_H_

#include <cmath>
#include <pybind11/pybind11.h>

#include "meta/index/ranker/ranker.h"

namespace pybind11
{
namespace detail
{
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

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)

void metapy_bind_index(pybind11::module& m);

#endif
