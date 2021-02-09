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

#include "graph_visualize.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

//-----------------------------------------------------------------------------
// GraphToPbTextFile
//-----------------------------------------------------------------------------
void GraphToPbTextFile(Graph* graph, const string& filename) {
  GraphDef g_def;
  graph->ToGraphDef(&g_def);

  string graph_pb_str;
  protobuf::TextFormat::PrintToString(g_def, &graph_pb_str);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  ostrm_out << graph_pb_str;
}

//-----------------------------------------------------------------------------
// GraphToDotFile
//-----------------------------------------------------------------------------
void GraphToDotFile(Graph* graph, const std::string& filename,
                    const std::string& title) {
  std::string dot = GraphToDot(graph, title);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  ostrm_out << dot;
}

static std::string color_string(unsigned int color) {
  std::stringstream ss;
  ss << "#" << std::setfill('0') << std::setw(6) << std::hex
     << (color & 0xFFFFFF);
  return ss.str();
}

//-----------------------------------------------------------------------------
// GraphToDot
//-----------------------------------------------------------------------------
std::string GraphToDot(Graph* graph, const std::string& title) {
  //
  // We'll be assigning distinct colors to the nodes in each found partition.
  //

  // Apparently these are called "Kelly's 22 colors of maximum contrast."
  static const int num_partition_colors = 22;
  static unsigned int partition_bg_colors[num_partition_colors]{
      0xF3C300, 0x875692, 0xF38400, 0xA1CAF1, 0xBE0032, 0xC2B280,
      0x848482, 0x008856, 0xE68FAC, 0x0067A5, 0xF99379, 0x604E97,
      0xF6A600, 0xB3446C, 0xDCD300, 0x882D17, 0x8DB600, 0x654522,
      0xE25822, 0x2B3D26, 0xF2F3F4, 0x222222};

  // We want to color text either black or white, depending on what works best
  // with the chosen background color.
  //
  // Algorithm found at:
  //
  //   https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  static auto make_fg_color = [](unsigned int bg_color) {
    unsigned int red = (bg_color & 0xFF0000) >> 16;
    unsigned int green = (bg_color & 0x00FF00) >> 8;
    unsigned int blue = (bg_color & 0x0000FF);

    if (red * 0.299 + green * 0.587 + blue * 0.114 > 186) {
      return 0x000000;
    } else {
      return 0xFFFFFF;
    }
  };

  int seen_partition_count = 0;
  std::map<int, unsigned int> partition_color_map;

  //
  // Emit preamble.
  //
  std::ostringstream dot_string;
  dot_string << "digraph G {\n";
  dot_string << "labelloc=\"t\";\n";
  dot_string << "label=<<b>TensorFlow Graph: " << title << "</b><br/><br/>>;\n";

  //
  // Emit each node.
  //
  for (auto id = 0; id < graph->num_node_ids(); ++id) {
    const Node* node = graph->FindNodeId(id);

    // Skip deleted nodes.
    if (node == nullptr) continue;

    // Skip source and sink nodes.
    if (!node->IsOp()) continue;

    // we will check if the nodes is under partition,
    // if the nodes have partition_id, then its
    // selected by dnnl_graph backend to be in partition
    bool is_partitioned;
    if (GetNodeAttr(node->attrs(), "_partition_id", &is_partitioned) !=
        Status::OK()) {
      is_partitioned = false;
    }

    // Decide on colors, style.
    unsigned int bg_color = 0xf2f2f2;
    string style;

    // partitioned nodes get a color based on their partition index.
    int partition_idx;
    if (GetNodeAttr(node->attrs(), "_partition_id", &partition_idx) ==
        Status::OK()) {
      if (partition_color_map.find(partition_idx) ==
          partition_color_map.end()) {
        bg_color = partition_bg_colors[seen_partition_count];
        partition_color_map[partition_idx] = bg_color;
        if (seen_partition_count < num_partition_colors - 1) {
          seen_partition_count++;
        }
      } else {
        bg_color = partition_color_map[partition_idx];
      }
      style = "filled";
    }
    // Nodes marked for partitioning get a solid border and an Intel blue
    // background.
    else if (is_partitioned) {
      style = "solid,filled";
      bg_color = 0x0071c5;
    }
    // Any other nodes are dashed.
    else {
      style = "dashed,filled";
    }

    unsigned int fg_color = make_fg_color(bg_color);
    unsigned int border_color = 0x000000;

    dot_string << "node_" << node;
    dot_string << " [label=<<b>" << node->type_string() << "</b><br/>";
    dot_string << node->name() << "<br/>";

    // Print the data type if this node an op node
    DataType datatype;
    if (GetNodeAttr(node->def(), "T", &datatype) == Status::OK()) {
      dot_string << DataTypeString(datatype) << "<br/>";
    }

    dot_string << ">";

    dot_string << ", shape=rect";
    dot_string << ", style=\"" << style << "\"";
    dot_string << ", fontcolor=\"" << color_string(fg_color) << "\"";
    dot_string << ", color=\"" << color_string(border_color) << "\"";
    dot_string << ", fillcolor=\"" << color_string(bg_color) << "\"";
    dot_string << " ];" << std::endl;

    //
    // Emit each of this node's input edges. (Graphviz does not care about
    // order here.)
    //
    for (const Edge* edge : node->in_edges()) {
      if (edge != nullptr) {
        const Node* src = edge->src();
        const Node* dst = edge->dst();

        // Skip edges from source and sink nodes.
        if (!src->IsOp()) continue;

        // Control edges are red, data edges are black.
        string arrow_color = edge->IsControlEdge() ? "#ff0000" : "#000000";

        dot_string << "node_" << src << " -> ";
        dot_string << "node_" << dst << " [color=\"" << arrow_color << "\"]\n";
      }
    }
  }

  //
  // Emit footer.
  //
  dot_string << "}\n";

  return dot_string.str();
}

bool DumpAllGraphs() {
  return std::getenv("TF_DNNL_GRAPH_DUMP_GRAPHS") != nullptr;
}

std::string DotFilename(std::string kind, int idx) {
  return GraphFilenamePrefix(kind, idx) + ".dot";
}

std::string DotFilename(std::string kind, int idx, int sub_idx) {
  return GraphFilenamePrefix(kind, idx, sub_idx) + ".dot";
}

std::string PbtxtFilename(std::string kind, int idx) {
  return GraphFilenamePrefix(kind, idx) + ".pbtxt";
}

std::string PbtxtFilename(std::string kind, int idx, int sub_idx) {
  return GraphFilenamePrefix(kind, idx, sub_idx) + ".pbtxt";
}

std::string GraphFilenamePrefix(std::string kind, int idx) {
  std::stringstream ss;
  ss << kind << "_" << std::setfill('0') << std::setw(4) << idx;
  return ss.str();
}

std::string GraphFilenamePrefix(std::string kind, int idx, int sub_idx) {
  std::stringstream ss;
  ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
     << std::setw(4) << sub_idx;
  return ss.str();
}

}  // namespace tf_dnnl_graph_bridge

}  // namespace tensorflow
