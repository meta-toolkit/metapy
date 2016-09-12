/**
 * @file metapy_embeddings.h
 * @author Chase Geigle
 */

#ifndef METAPY_EMBEDDINGS_H_
#define METAPY_EMBEDDINGS_H_

#include <cmath>
#include <pybind11/pybind11.h>

void metapy_bind_embeddings(pybind11::module& m);

#endif
