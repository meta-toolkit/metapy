/**
 * @file metapy_learn.h
 * @author Chase Geigle
 */

#ifndef METAPY_LEARN_H_
#define METAPY_LEARN_H_

#include <cmath>
#include <pybind11/pybind11.h>

template <class DatasetView>
DatasetView make_sliced_dataset_view(const DatasetView& dv,
                                     pybind11::slice slice)
{
    std::size_t start, stop, step, slicelength;
    if (!slice.compute(dv.size(), &start, &stop, &step, &slicelength))
        throw pybind11::error_already_set{};

    std::vector<std::size_t> indices(slicelength);
    auto it = dv.begin() + start;
    for (std::size_t i = 0; i < slicelength; ++i)
    {
        indices[i] = it->id;
        it += step;
    }

    return DatasetView{dv, std::move(indices)};
}

void metapy_bind_learn(pybind11::module& m);

#endif
