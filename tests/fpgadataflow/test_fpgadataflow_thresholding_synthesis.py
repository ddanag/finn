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

import pytest
import numpy as np
import os, subprocess
from onnx import TensorProto, helper
import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.util.gdrive import *
from finn.util.basic import pynq_part_map
from finn.util.basic import gen_finn_dt_tensor
from finn.analysis.fpgadataflow.get_timing import get_timing
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.custom_op.registry import getCustomOp
from finn.custom_op.general.multithreshold import multithreshold
from finn.transformation.streamline import Streamline
from finn.transformation.general import GiveUniqueNodeNames
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.cleanup import CleanUp
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition

BOARD = "ZCU104"
FPGA = pynq_part_map[BOARD]
TARGET_CLK_PERIOD = 5
WORKSHEET_NAME = 'Thresholding_layer_resources'
#WORKSHEET_NAME = 'Thresholding_layer_resources_test_set'

def make_single_thresholding_modelwrapper(T, pe, idt, odt, actval, mem_mode, ram_style):
    NumChannels = T.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, NumChannels])

    node_inp_list = ["inp", "thresh"]

    Thresholding_node = helper.make_node(
        "Thresholding_Batch",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        PE=pe,
        numSteps=T.shape[1],
        inputDataType=idt.name,
        weightDataType=idt.name,  # will be set by MinimizeAccumulatorWidth
        outputDataType=odt.name,
        ActVal=actval,
        mem_mode=mem_mode,
        ram_style=ram_style,
    )
    graph = helper.make_graph(
        nodes=[Thresholding_node],
        name="thresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("thresh", idt)
    model.set_initializer("thresh", T)
    return model

headers = ["FPGA", "ich", "nf", "pe", "idt", "act", "mem_mode", "ram_style", "Resources from:",
            "LUT", "LUTRAM", "FF", "SRL", "DSP", "BRAM", "BRAM_18K", "BRAM_36K", "Total_BRAM_18K", "BRAM_efficiency", "URAM", 
            "URAM_efficiency", "Carry", "TargetClockPeriod", "EstimatedClockPeriod", "Delay", "TargetClockFrequency [MHz]",
            "EstimatedClockFrequency [MHz]", "finn_commit", "vivado_version", "vivado_build_no"]
config_headers = ["FPGA", "ich", "nf", "pe", "idt", "act", "mem_mode", "ram_style", "Resources from:", "finn_commit", "vivado_version", "vivado_build_no"]

def upload_data_to_thresholding_dashboard(test_parameters, resources):
    
    data_dict = {key: -1 for key in headers} 
    
    for header in headers:
        for key in test_parameters.keys():
            if header == key:
                data_dict[header] = str(test_parameters[key])
        
        if header == "Resources from:":
            data_dict[header] = resources[0]
            data_dict["Total_BRAM_18K"] = 0
            data_dict["DSP"] = 0

        res = eval(resources[1])
        for key in res.keys(): 
            if header == key and header != "DSP":
                data_dict[header] = res[key]

                if key == "BRAM_18K":
                    data_dict["Total_BRAM_18K"] += int(res[key])
                elif key == "BRAM_36K":
                    data_dict["Total_BRAM_18K"] += 2* int(res[key])

            elif "DSP" in key and "DSP" in header:
                data_dict["DSP"] += int(res[key])

        finn_commit_dict = resources[2]
        data_dict["finn_commit"] = finn_commit_dict["finn_commit"]
        if resources[0] == "hls":
            data_dict["vivado_version"] = finn_commit_dict["vivado_version"]
            data_dict["vivado_build_no"] = finn_commit_dict["vivado_build_no"]

        if resources[0] == "synthesis":
            data_dict["EstimatedClockPeriod"] = float(data_dict["TargetClockPeriod"]) - float(data_dict["Delay"])
        elif len(resources) > 3:
            timing = resources[3]
            for key in timing.keys(): 
                if header == key:
                    data_dict[header] = timing[key]
                    data_dict["Delay"] = float(data_dict["TargetClockPeriod"]) - float(timing[key])

    #compute clock freq
    data_dict["TargetClockFrequency [MHz]"] = 1/float(data_dict["TargetClockPeriod"]) * 1000
    if data_dict["EstimatedClockPeriod"] != -1:
        data_dict["EstimatedClockFrequency [MHz]"] = 1/float(data_dict["EstimatedClockPeriod"]) * 1000
    
    data_dict['vivado_build_no'] = int(data_dict['vivado_build_no'])

    #check if the configuration already exists in the worksheet
    config_dict = {key: data_dict[key] for key in config_headers}
    matched, row_index = search_in_resource_dashboard(WORKSHEET_NAME, config_dict)
    overwrite = matched

    upload_to_resource_dashboard(WORKSHEET_NAME, data_dict, overwrite, row_index)

# activation: None or DataType
@pytest.mark.parametrize("act", [DataType.BIPOLAR, DataType.INT2, DataType.INT3, DataType.INT4])
#additional configs for resource modelling test set
#@pytest.mark.parametrize("act", [DataType.INT5, DataType.INT7, DataType.INT8])
# input datatype
@pytest.mark.parametrize("idt", [DataType.UINT12, DataType.UINT16, DataType.UINT20, DataType.UINT24, DataType.UINT28, DataType.UINT32])
#additional configs for resource modelling test set
#@pytest.mark.parametrize("idt", [DataType.UINT14, DataType.UINT18, DataType.UINT22, DataType.UINT26, DataType.UINT30])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 1, 2, 4, 8])
# number of input features
@pytest.mark.parametrize("ich", [16, 32, 64, 96, 128, 192, 256])
#additional configs for resource modelling test set
#@pytest.mark.parametrize("ich", [48, 80, 160, 320])
# memory mode
@pytest.mark.parametrize("mem_mode", ["decoupled"])
# ram style
@pytest.mark.parametrize("ram_style", ["distributed", "block"])
# Upload to google spreadsheet
@pytest.mark.parametrize("upload", [True])
# Remove artefacts
@pytest.mark.parametrize("cleanup", [True])
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.resource_estimation
def test_fpgadataflow_thresholding_synthesis(idt, act, nf, ich, mem_mode, ram_style, upload, cleanup):
    if nf == -1:
        nf = ich
    pe = ich // nf
    assert ich % pe == 0

    odt = act

    if odt == DataType.BIPOLAR:
        actval = 0
    else:
        actval = odt.min()
    
    #generate thresholds
    n_steps = act.get_num_possible_values() - 1
    if act == DataType.BIPOLAR:
        T = np.random.randint((1<<(idt.bitwidth()-1)), idt.max() + 1, (ich, n_steps)).astype(np.float32)
        # make the vivado_hls threshold bug appear (incorrect rtlsim result when first
        # threshold of first channel is zero, while using BIPOLAR output)
        #if act == DataType.BIPOLAR:
        #    T[0][0] = 0
    else:
        T = np.zeros((ich, n_steps))
        j = 0
        reverse = 0
        for i in range(ich):   
            T[i][j] = np.random.randint((1<<(idt.bitwidth()-1)), idt.max())
            k_set = j
            if reverse:
                j = j - 1
            else:
                j = j + 1
            if j == n_steps-1:
                reverse = 1
            elif j == 0:
                reverse = 0 
            for k in range(n_steps):
                if k < k_set:
                    T[i][k] = np.random.randint(idt.min(), (T[i][k_set]-1))
                elif k > k_set:
                    T[i][k] = np.random.randint(T[i][k_set], idt.max())

        T = T.astype(np.float32)
    #provide non-decreasing thresholds
    T = np.sort(T, axis=1)

    model = make_single_thresholding_modelwrapper(T, pe, idt, odt, actval, mem_mode, ram_style)
    
    ###
    #create_worksheet_in_resource_dashboard(WORKSHEET_NAME, 10, headers)
    #import pdb; pdb.set_trace()

    #get_weightstream_width and check if it's under the HLS limitation (32768)
    node = model.get_nodes_by_op_type("Thresholding_Batch")[0]
    node = getCustomOp(node)
    weightstream_width = node.get_weightstream_width()
    
    if weightstream_width <= 32768:
        #Streamlining
        model = model.transform(Streamline())

        #Convert to HLS layers
        model = model.transform(to_hls.InferThresholdingLayer(mem_mode))

        #Dataflow Partitioning
        model = model.transform(CreateDataflowPartition())

        #get the StreamingDataflowPartition
        sdp_node = model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")

        #save the dataflow partition with a different name for easier access
        dataflow_model = ModelWrapper(dataflow_model_filename)

        dataflow_model = dataflow_model.transform(GiveUniqueNodeNames())

        dataflow_model.save("thresholding_layer_before_synth.onnx")

        test_parameters = {'FPGA': FPGA, 'ich': ich, 'nf': nf, 'pe': pe, 'idt': idt, 'act': act, 'mem_mode': mem_mode, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD}
        finn_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace/finn")
        finn_commit = finn_commit.decode("utf-8").strip()

        #PrepareIP, #HLSSynthIP
        dataflow_model = dataflow_model.transform(PrepareIP(FPGA, TARGET_CLK_PERIOD))
        dataflow_model = dataflow_model.transform(HLSSynthIP())

        #skip out of context synthesis if config_dict already exists in finn-resource-dashboard
        config_dict = {'FPGA': FPGA, 'ich': ich, 'nf': nf, 'pe': pe, 'idt': idt, 'act': act, 'mem_mode': mem_mode, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD, 'Resources from:': 'synthesis'}
        
        ###have to get this from vivado - tcl script
        vivado_version = '2020.1'
        vivado_build_no = '2902540'

        if upload:
            matched, row_index = search_in_resource_dashboard(WORKSHEET_NAME, config_dict)
        else:
            matched = False
            
        if not matched:
            #CreateStitchedIP, OutOfContextSynth
            dataflow_model = dataflow_model.transform(CreateStitchedIP(FPGA, TARGET_CLK_PERIOD))
            dataflow_model = dataflow_model.transform(SynthOutOfContext(part = FPGA, clk_period_ns = TARGET_CLK_PERIOD))

            synthesis_resources = ["synthesis"]
            ret = dataflow_model.get_metadata_prop("res_total_ooc_synth")
            synthesis_resources.append(ret)
            synthesis_resources.append({'finn_commit': finn_commit})

            ret = eval(ret)
            vivado_version = ret['vivado_version']
            vivado_build_no = ret['vivado_build_no']

            if upload:
                upload_data_to_thresholding_dashboard(test_parameters, synthesis_resources)

            dataflow_model.save("thresholding_model_after_ooosynth.onnx")
        
        #skip getting hls estimates if config_dict already exists in finn-resource-dashboard
        config_dict = {'FPGA': FPGA, 'ich': ich, 'nf': nf, 'pe': pe, 'idt': idt, 'act': act, 'mem_mode': mem_mode, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD, 'Resources from:': 'hls'}

        if upload:
            matched, row_index = search_in_resource_dashboard(WORKSHEET_NAME, config_dict)
        else:
            matched = False
            
        if not matched:       
            #get resources estimated by hls
            dataflow_model_hls = dataflow_model.transform(AnnotateResources(mode="hls"))
            dataflow_model_hls.save("thresholding_model_hls.onnx")

            hls_resources = ["hls"]
            custom_ops_hls = getCustomOp(dataflow_model_hls.graph.node[0])
            hls_resources.append(custom_ops_hls.get_nodeattr("res_hls"))
            hls_resources.append({'finn_commit': finn_commit, 'vivado_version': vivado_version, 'vivado_build_no': vivado_build_no})
            
            #get EstimatedClockPeriod
            timing = get_timing(dataflow_model_hls, "hls")
            hls_resources.append(timing[dataflow_model_hls.graph.node[0].name])
            
            if upload:
                upload_data_to_thresholding_dashboard(test_parameters, hls_resources)
            
            if cleanup:
                dataflow_model_hls.transform(CleanUp())

        #skip getting finn estimates if config_dict already exists in finn-resource-dashboard
        config_dict = {'FPGA': FPGA, 'ich': ich, 'nf': nf, 'pe': pe, 'idt': idt, 'act': act, 'mem_mode': mem_mode, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD, 'Resources from:': 'estimate'}

        if upload:
            matched, row_index = search_in_resource_dashboard(WORKSHEET_NAME, config_dict)
        else:
            matched = False
            
        if not matched:  
            #get estimated resources
            dataflow_model_estimate = dataflow_model.transform(AnnotateResources(mode="estimate"))
            dataflow_model_estimate.save("thresholding_model_estimate.onnx")

            estimate_resources = ["estimate"]
            custom_ops_estimate = getCustomOp(dataflow_model_estimate.graph.node[0])
            estimate_resources.append(custom_ops_estimate.get_nodeattr("res_estimate"))
            estimate_resources.append({'finn_commit': finn_commit})

            if upload:
                upload_data_to_thresholding_dashboard(test_parameters, estimate_resources)
            if cleanup:
                dataflow_model_estimate.transform(CleanUp())

        if cleanup:
            model.transform(CleanUp())
            dataflow_model.transform(CleanUp())