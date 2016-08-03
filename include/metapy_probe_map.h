/**
 * @file metapy_probe_map.h
 * @author Chase Geigle
 *
 * Provides a caster from hashing::probe_map<K, V> for pybind11.
 */

#ifndef METAPY_PROBE_MAP_H_
#define METAPY_PROBE_MAP_H_

#include <cmath>
#include <pybind11/pybind11.h>

#include "meta/hashing/probe_map.h"

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
                                   + value_conv::name() + _(">"));
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
}
}

#endif
