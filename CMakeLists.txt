cmake_minimum_required(VERSION 3.2.0)

project(metapy)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(METAPY_PYTHON_VERSION "" CACHE STRING
    "Python version to use for compiling the extension")

add_subdirectory(deps/meta EXCLUDE_FROM_ALL)

if (NOT PYTHON_INCLUDE_DIRS)
  if (NOT ${METAPY_PYTHON_VERSION} STREQUAL "")
    list(APPEND Python_ADDITIONAL_VERSIONS ${METAPY_PYTHON_VERSION})
    find_package(PythonLibs ${METAPY_PYTHON_VERSION} EXACT)
    if (NOT PythonLibs_FOUND)
      find_package(PythonLibs ${METAPY_PYTHON_VERSION} REQUIRED)
    endif()
  else()
    find_package(PythonLibs REQUIRED)
  endif()
else()
  message("-- Using manual Python include dirs: ${PYTHON_INCLUDE_DIRS}")
endif()

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/pybind11/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(metapy SHARED src/metapy_analyzers.cpp
                          src/metapy_index.cpp
                          src/metapy_sequence.cpp
                          src/metapy_parser.cpp
                          src/metapy_embeddings.cpp
                          src/metapy.cpp)
target_link_libraries(metapy meta-index meta-ranker meta-sequence
                      meta-greedy-tagger meta-parser meta-embeddings)

# don't add a "lib" prefix to the metapy shared library
set_target_properties(metapy PROPERTIES PREFIX "")

if (APPLE)
  # OS X stupid fixes
  # (see http://pybind11.readthedocs.org/en/latest/cmake.html)
  set_target_properties(metapy PROPERTIES
                        MACOSX_RPATH "."
                        LINK_FLAGS "-undefined dynamic_lookup "
                        SUFFIX ".so")
endif()

if (WIN32)
  set_target_properties(metapy PROPERTIES SUFFIX ".pyd")
  target_link_libraries(metapy ${PYTHON_LIBRARY})
  target_compile_definitions(metapy PUBLIC -DMS_WIN64)

  # fix for std::_hypot has not been declared
  target_compile_definitions(metapy PUBLIC -D_hypot=hypot)
endif()

install(TARGETS metapy DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/dist/metapy)
