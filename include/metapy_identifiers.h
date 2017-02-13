/**
 * @file metapy_identifiers.h
 * @author Chase Geigle
 *
 * Provides a caster to/from meta::util::identifier<Tag, T> for pybind11.
 */

#ifndef METAPY_IDENTIFIERS_H_
#define METAPY_IDENTIFIERS_H_

#include <cmath>
#include <pybind11/pybind11.h>

#include "meta/meta.h"

namespace pybind11
{
namespace detail
{

template <class Type>
struct identifier_caster
{
    using underlying_type = typename Type::underlying_type;

    PYBIND11_TYPE_CASTER(Type, _("id"));

    bool load(handle src, bool convert)
    {
        make_caster<underlying_type> conv;
        if (!conv.load(src, convert))
            return false;
        value = Type{(underlying_type)conv};
        return true;
    }

    static handle cast(const Type& src, return_value_policy policy,
                       handle parent)
    {
        return make_caster<underlying_type>::cast(
            static_cast<underlying_type>(src), policy, parent);
    }
};

template <class Tag, class T>
struct type_caster<meta::util::numerical_identifier<Tag, T>>
    : identifier_caster<meta::util::numerical_identifier<Tag, T>>
{
};

template <class Tag, class T>
struct type_caster<meta::util::identifier<Tag, T>>
    : identifier_caster<meta::util::identifier<Tag, T>>
{
};
}
}
#endif
