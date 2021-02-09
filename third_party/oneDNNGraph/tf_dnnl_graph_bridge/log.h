/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef TF_DNNL_GRAPH_BRIDGE_LOG_H_
#define TF_DNNL_GRAPH_BRIDGE_LOG_H_

#include <string>

#include "common.h"

class LlgaLogMessage : public tensorflow::internal::LogMessage {
 public:
  static tensorflow::int64 MinLlgaVLogLevel();
};

#define DNNL_GRAPH_VLOG_IS_ON(lvl) ((lvl) <= LlgaLogMessage::MinLlgaVLogLevel())

#define DNNL_GRAPH_VLOG(lvl)      \
  if (DNNL_GRAPH_VLOG_IS_ON(lvl)) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)

#endif  // TF_DNNL_GRAPH_BRIDGE_LOG_H_
