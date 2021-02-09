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

set(LIBTF_DNNL_GRAPH tf_dnnl_graph_bridge)

include_directories(${TensorFlow_INCLUDE_DIR})
message(STATUS "TF_DNNL_GRAPH_bridge: TensorFlow: ${TensorFlow_INCLUDE_DIR}")
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)
include_directories(${ABSL_INCLUDE_DIR})

set(DNNL_GRAPH_DEVICE_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason the following is needed for ABI compatibility with TF.
# (There must be some dependency on it in the struct/class definitions.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
set(TF_DNNL_GRAPH_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(SRC
   #allocator.cc
   runtime.cc
   dnnl_graph_fused_op_register.cc
   dnnl_graph_fused_op.cc
   replace_fusions.cc
   graph_visualize.cc
   log.cc
   partitioner.cc
   utils.cc
   version.cc
   add_identityn.cc
   translate_op.cc
)

add_library(${LIBTF_DNNL_GRAPH} SHARED ${SRC})
set_target_properties( ${LIBTF_DNNL_GRAPH}
  PROPERTIES
  DIRECTORY ${TF_DNNL_GRAPH_DIR}
  )
target_link_libraries(
  ${LIBTF_DNNL_GRAPH} 
  ${TensorFlow_FRAMEWORK_LIBRARY} 
  dnnl_graph_lib
  dl
  pthread
  absl_algorithm
  absl_container
  absl_strings
  )

add_definitions(-DTF_DNNL_GRAPH_USE_GRAPPLER_OPTIMIZER)
target_compile_definitions( 
    ${LIBTF_DNNL_GRAPH} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}
)

message(STATUS "TensorFlow_CXX_ABI from tf_dnnl_graph_build: ${TensorFlow_CXX_ABI}")
target_include_directories(${LIBTF_DNNL_GRAPH} PUBLIC "${DNNL_GRAPH_INCLUDE_DIR}")
message(STATUS "for TF_DNNL_GRAPH : DNNL_GRAPH_INCLUDE_DIR: ${DNNL_GRAPH_INCLUDE_DIR}")

#------------------------------------------------------------------------------
#installation 
#------------------------------------------------------------------------------
message(STATUS "TF_DNNL_GRAPH_INSTALL_LIB_DIR: ${INSTALL_LIB_DIR}")
message(STATUS "TF_DNNL_GRAPH_INSTALL_INCLUDE_DIR: ${INSTALL_INCLUDE_DIR}")

# First install the libtf_dnnl_graph_bridge.so and headers
# why are the headers NOT isntalling? TBD-swapna  same with logger
install(TARGETS ${LIBTF_DNNL_GRAPH}
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" )


#Now create a python module for tf_dnnl_graph_bridge
# ref to src and binary dir
install(CODE "set(CMAKE_SOURCE_DIR \"${CMAKE_SOURCE_DIR}\")")
install(CODE "set(CMAKE_BINARY_DIR \"${CMAKE_BINARY_DIR}\")")
# ref to install paths
install(CODE "set(INSTALL_LIB_DIR \"${INSTALL_LIB_DIR}\")")
install(CODE "set(INSTALL_INCLUDE_DIR \"${INSTALL_INCLUDE_DIR}\")")
install(CODE "set(TF_DNNL_GRAPH_ENV_PATH \"${TF_DNNL_GRAPH_ENV_PATH}\")")
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../python/CreateModule.cmake")
