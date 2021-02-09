# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

message(STATUS "script: CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "script: CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "script: install_INCLUDE_DIR: ${INSTALL_INCLUDE_DIR}")
message(STATUS "script: install_lib_DIR: ${INSTALL_LIB_DIR}")
message(STATUS "script: TF_DNNL_GRAPH_ENV_PATH: ${TF_DNNL_GRAPH_ENV_PATH}")

set(INIT_PY_IN  "${CMAKE_SOURCE_DIR}/python/tf_dnnl_graph_bridge/__init__.in.py")
set(INIT_PY     "${CMAKE_BINARY_DIR}/python/tf_dnnl_graph_bridge/__init__.py")

# Create the python/tf_dnnl_graph_bridge directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/tf_dnnl_graph_bridge)
set(INSTALL_PATH "${CMAKE_BINARY_DIR}/python/tf_dnnl_graph_bridge")
message(STATUS "install_path: ${INSTALL_PATH}")

file(COPY ${INSTALL_INCLUDE_DIR} DESTINATION ${INSTALL_PATH})

# dnnl_graph copies cmake lib into it's install -workaround
file(COPY ${INSTALL_LIB_DIR} DESTINATION ${INSTALL_PATH}
  FILES_MATCHING PATTERN "*.so*")

configure_file(${INIT_PY_IN} ${INIT_PY})

file(COPY ${INIT_PY} DESTINATION ${INSTALL_PATH})

set(SITE_PATH "${TF_DNNL_GRAPH_ENV_PATH}/lib/python3.6/site-packages/tf_dnnl_graph_bridge")
message("SITE_PATH: ${SITE_PATH}")

file(COPY "${INSTALL_PATH}/lib/" DESTINATION ${SITE_PATH}
  FILES_MATCHING PATTERN "*.so*")
file(COPY "${INSTALL_PATH}/include" DESTINATION ${SITE_PATH})
file(COPY "${INSTALL_PATH}/__init__.py" DESTINATION ${SITE_PATH})





