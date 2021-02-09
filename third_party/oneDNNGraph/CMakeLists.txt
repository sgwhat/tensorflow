# Copyright 2018-2020 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project (dnnl_graph_tensorflow_bridge CXX)

if (NOT DEFINED TF_VERSION)
  set(TF_VERSION 2.2.1)
endif()
message(STATUS "TF_VERSION: ${TF_VERSION}")

set(TF_TEMP ${TF_VERSION})
string(REPLACE "." ";" TF_VERSION_LIST ${TF_TEMP})
message(STATUS "LIST: ${TF_VERSION_LIST}")
list(GET TF_VERSION_LIST 0 TF_MAJOR_VERSION)
list(GET TF_VERSION_LIST 1 TF_MINOR_VERSION)
list(GET TF_VERSION_LIST 2 TF_PATCH_VERSION)

if(TF_MAJOR_VERSION GREATER_EQUAL 2)
  set(TENSORFLOW_STRING "tensorflow-cpu")
elseif(TF_MAJOR_VERSION EQUAL 1)
  set(TENSORFLOW_STRING "tensorflow")
endif()

set(REQ_FILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/cmake/requirements.txt.in)
set(REQ_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/cmake/requirements.txt)
configure_file(${REQ_FILE_IN} ${REQ_FILENAME})


# get all arguments TBD-swapna
# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(ExternalProject)

find_package (Python3 COMPONENTS Interpreter)
if(${Python3_Interpreter_FOUND})
  message(STATUS "Found python3")
else()
  message(ERROR "Could not find python3")
endif()

set(TF_DNNL_GRAPH_SOURCE_DIR "${CMAKE_SOURCE_DIR}/tf_dnnl_graph_bridge")

# Since UNIX support Bash we can use a Bash script to do the clang-format functions
# This is much faster than the cmake method
set(FORMAT_DIRS
    ${TF_DNNL_GRAPH_SOURCE_DIR}
)
if (UNIX)
    add_custom_target(style-check
        COMMAND
            ${CMAKE_SOURCE_DIR}/tools/check_code_format.sh ${FORMAT_DIRS}
    )
    add_custom_target(style-apply
        COMMAND
            ${CMAKE_SOURCE_DIR}/tools/apply_code_format.sh ${FORMAT_DIRS}
    )
    add_custom_target(style
        COMMAND
            ${CMAKE_SOURCE_DIR}/tools/apply_code_format.sh ${FORMAT_DIRS}
    )
else()
    add_custom_target(style-check
        COMMAND ${CMAKE_COMMAND}
        -DFORMAT_DIRS="${TF_DNNL_GRAPH_SOURCE_DIR}"
        -P ${CMAKE_SOURCE_DIR}/cmake/style_check.cmake
    )

    add_custom_target(style-apply
        COMMAND ${CMAKE_COMMAND}
        -DFORMAT_DIRS="${TF_DNNL_GRAPH_SOURCE_DIR}"
        -P ${CMAKE_SOURCE_DIR}/cmake/style_apply.cmake
    )

    add_custom_target(style
        COMMAND ${CMAKE_COMMAND}
        -DFORMAT_DIRS="${TF_DNNL_GRAPH_SOURCE_DIR}"
        -P ${CMAKE_SOURCE_DIR}/cmake/style_apply.cmake
    )
endif()

message(STATUS "CURRENT_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_SRC_DIR ${CMAKE_SOURCE_DIR}")

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
message(STATUS "BASEPATH ${CMAKE_SOURCE_DIR}")

INCLUDE_DIRECTORIES("${BASEPATH}")

# setup env
# create evn with script & set relative path for python
set(TF_DNNL_GRAPH_ENV_PATH "${CMAKE_BINARY_DIR}/tfenv")
message("Building a virtual env in ${TF_DNNL_GRAPH_ENV_PATH}...")

# Generate the virtualenv and ensure it's up to date.
execute_process( COMMAND ${Python3_EXECUTABLE} -m venv ${TF_DNNL_GRAPH_ENV_PATH} )

set(TF_DNNL_GRAPH_PYTHON ${TF_DNNL_GRAPH_ENV_PATH}/bin/python3)
message(STATUS "Python: ${TF_DNNL_GRAPH_PYTHON}")

execute_process( COMMAND ${TF_DNNL_GRAPH_ENV_PATH}/bin/pip install --upgrade pip)
execute_process( COMMAND ${TF_DNNL_GRAPH_ENV_PATH}/bin/pip install -r ${REQ_FILENAME} )

set(CMAKE_MACOSX_RPATH 1)
if(DEFINED TF_DNNL_GRAPH_RPATH)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow:${TF_DNNL_GRAPH_RPATH}")
else()
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow")
endif()

find_package(TensorFlow REQUIRED)
unset(TF_VERSION CACHE)

# move to dependent cmake dir- TBD-swapna
set(LIBDNNL_GRAPH "libdnnl_graph.so.0.1")


set(DNNL_GRAPH_TARGET_ARCH native CACHE STRING "Target CPU architecture to build for.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${DNNL_GRAPH_TARGET_ARCH}")

set(DNNL_GRAPH_TUNE_ARCH native CACHE STRING "Target CPU architecture to build for.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=${DNNL_GRAPH_TARGET_ARCH}")

message(STATUS "USE_PRE_BUILT_DNNL_GRAPH:         ${USE_PRE_BUILT_DNNL_GRAPH}")
message(STATUS "DNNL_GRAPH_TARGET_ARCH:           ${DNNL_GRAPH_TARGET_ARCH}")
message(STATUS "DNNL_GRAPH_TUNE_ARCH:             ${DNNL_GRAPH_TUNE_ARCH}")

include(ProcessorCount)
ProcessorCount(NUM_MAKE_PROCESSES)

# hiost this to the top and name it nice TBD-swapna
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
message(STATUS "INSTALL_LIB_DIR: ${INSTALL_LIB_DIR}")
message(STATUS "INSTALL_INCLUDE_DIR: ${INSTALL_INCLUDE_DIR}")

# Find and build dnnl_graph - if not using pre-built one
ExternalProject_Add(
  ext_dnnl_graph
  GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
  GIT_TAG dev-graph
  CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DDNNL_GRAPH_TARGET_ARCH=${DNNL_GRAPH_TARGET_ARCH}
  -DDNNL_GRAPH_TUNE_ARCH=${DCMAKE_INSTALL_PREFIX}
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/stamp"
  DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/download"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/build"
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j ${NUM_MAKE_PROCESSES} dnnl_graph
  STEP_TARGETS build
  EXCLUDE_FROM_ALL TRUE
  )


set(DNNL_GRAPH_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/src")
set(DNNL_GRAPH_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/llga/build/src")
set(DNNL_GRAPH_INCLUDE_DIR "${DNNL_GRAPH_DIR}/include")

add_library(dnnl_graph_lib INTERFACE)
# only doing build of dnnl_graph - NO install ( messes with our make)
add_dependencies(dnnl_graph_lib ext_dnnl_graph-build)
target_include_directories(dnnl_graph_lib SYSTEM INTERFACE ${DNNL_GRAPH_INCLUDE_DIR})
# ask Sang Ik what's wrong here TBD-swapna
target_link_libraries(dnnl_graph_lib INTERFACE
  ${DNNL_GRAPH_LIB_DIR}/${LIBDNNL_GRAPH}
  )


# Add the directories to be built
add_subdirectory(third-party)
# this needs to be last? TBD-swapna
add_subdirectory(tf_dnnl_graph_bridge)

#------------------------------------------------------------------------------
#installation of dnnl_graph & it's dependencies - this needs to move- TBD-swapna
#------------------------------------------------------------------------------
install(DIRECTORY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/install/lib")
#this should be defined too? TBD-swapna
#install(DIRECTORY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/install/include")

install(DIRECTORY ${DNNL_GRAPH_INCLUDE_DIR}/ DESTINATION ${INSTALL_INCLUDE_DIR})
#  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(FILES ${DNNL_GRAPH_LIB_DIR}/${LIBDNNL_GRAPH} DESTINATION ${INSTALL_LIB_DIR})





