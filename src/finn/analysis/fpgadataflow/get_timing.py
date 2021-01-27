# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import os
import pandas as pd
import xml.etree.ElementTree as ET
import finn.custom_op.registry as registry
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp

def get_timing(model, mode):

    timing_dict = {}
    timing_dict[mode] = dict()
    
    if mode == "hls":

        for node in model.graph.node:
            if is_fpgadataflow_node(node) is True:
                timing_dict[mode][node.name] = dict() 
                timing_dict[mode][node.name]["EstimatedClockPeriod"] = 0
                
                inst = registry.getCustomOp(node)
                code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
                
                if code_gen_dir == "":
                    warnings.warn(
                        """Could not find report files, values will be set to zero
                        for this node. Please run "PrepareIP" transformation and
                        "HLSSynthIP" first to generate the report files"""
                    )
                else:
                    xmlfile = "{}/project_{}/sol1/syn/report/{}_csynth.xml".format(code_gen_dir, node.name, node.name)
                    
                    if os.path.isfile(xmlfile):
                        tree = ET.parse(xmlfile)
                        root = tree.getroot()
                        for item in root.findall("PerformanceEstimates/SummaryOfTimingAnalysis"):
                            for child in item:
                                if child.tag == "EstimatedClockPeriod":
                                    timing_dict[mode][node.name][child.tag] = child.text
                    else:
                        warnings.warn(
                            """Could not find report files, values will be set to zero
                            for this node. Please run "PrepareIP" transformation and
                            "HLSSynthIP" first to generate the report files"""
                        )

        return timing_dict[mode]
        
    elif mode == "synth":

        timing_dict["synth"]["Delay"] = 0
        vivado_pynq_proj_dir =  model.get_metadata_prop("vivado_pynq_proj")

        if vivado_pynq_proj_dir == "":
            warnings.warn("""Could not find report files. Please run synthesis first""")
        else:
            timing_file = "{}/finn_zynq_link.runs/impl_1/top_wrapper_timing_summary_postroute_physopted.rpt".format(vivado_pynq_proj_dir)

        if os.path.isfile(timing_file):
            df = pd.read_csv(timing_file, sep='delimiter', header=None)
            index_row = df[df.apply(lambda r: r.str.contains('Max Delay Paths', case=False).any(), axis=1)].index.values.astype(int)
            df = df[index_row[0]::]
            df = df[df.apply(lambda row: row.str.contains('Slack').any(), axis=1)]
            row = df.iloc[0]
            row = row[0].split(' ')
            for item in row:
                for character in item:
                    if character.isdigit():
                        delay = item[0:-2]
            timing_dict["synth"]["Delay"] = delay
        else:
            warnings.warn("""Could not find report files. Please run synthesis first""")

        return timing_dict[mode]

    else:
        warnings.warn("""Unrecognized mode: should be 'hls' or 'synth'""")