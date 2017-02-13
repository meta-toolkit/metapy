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
#include "meta/util/optional.h"
#include "metapy_identifiers.h"

namespace pybind11
{
namespace detail
{
// add conversion for meta::index::search_result
// see: pybind11/cast.h
template <>
struct type_caster<meta::index::search_result>
{
    using type = std::pair<meta::doc_id, float>;

    // Python -> C++
    bool load(handle src, bool convert)
    {
        value = meta::util::nullopt;
        make_caster<type> conv;
        if (!conv.load(src, convert))
            return false;

        auto pr = static_cast<type>(conv);
        value = meta::index::search_result{pr.first, pr.second};
        return true;
    }

    // C++ -> Python
    static handle cast(const meta::index::search_result& sr,
                       return_value_policy policy, handle parent)
    {
        auto o1 = reinterpret_steal<object>(
            make_caster<type::first_type>::cast(sr.d_id, policy, parent));
        auto o2 = reinterpret_steal<object>(
            make_caster<type::second_type>::cast(sr.score, policy, parent));

        if (!o1 || !o2)
            return handle();

        tuple result(2);
        PyTuple_SET_ITEM(result.ptr(), 0, o1.release().ptr());
        PyTuple_SET_ITEM(result.ptr(), 1, o2.release().ptr());
        return result.release();
    }

    static PYBIND11_DESCR name()
    {
        return type_descr(_("SearchResult"));
    }

    static handle cast(const meta::index::search_result* sr,
                       return_value_policy policy, handle parent)
    {
        return cast(*sr, policy, parent);
    }

    operator meta::index::search_result*()
    {
        return &*value;
    }

    operator meta::index::search_result&()
    {
        return *value;
    }

    template <class T>
    using cast_op_type = pybind11::detail::cast_op_type<T>;

  protected:
    meta::util::optional<meta::index::search_result> value;
};
}
}

void metapy_bind_index(pybind11::module& m);

#endif
