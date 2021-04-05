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
from onnx import TensorProto, helper

from finn.custom_op.registry import getCustomOp
import finn.core.onnx_exec as oxe
import finn.custom_op.general.xnorpopcount as xp
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.general.multithreshold import multithreshold
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.general import GiveReadableTensorNames
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import calculate_signed_dot_prod_range, gen_finn_dt_tensor
from finn.util.basic import pynq_part_map
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.infer_shapes import InferShapes
from finn.analysis.fpgadataflow.get_timing import get_timing
from finn.util.onnx import nchw_to_nhwc

BOARD = "U250"
FPGA = alveo_part_map[BOARD]
TARGET_CLK_PERIOD = 5
WORKSHEET_NAME = 'Sliding_Window_layer_resources'

def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )

    im2col_node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        backend="fpgadataflow",
        stride=stride,
        kernel_size=k,
        input_shape=str((1, ifm_dim, ifm_dim, ifm_ch)),
        pad_amount=0,
        pad_value=0,
    )
    graph = helper.make_graph(
        nodes=[im2col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model

def make_single_slidingwindow_modelwrapper(
    k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt, dw=0
):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )

    SlidingWindow_node = helper.make_node(
        "ConvolutionInputGenerator",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ConvKernelDim=k,
        IFMChannels=ifm_ch,
        IFMDim=ifm_dim,
        OFMDim=ofm_dim,
        SIMD=simd,
        Stride=stride,
        inputDataType=idt.name,
        outputDataType=odt.name,
        depthwise=dw,
    )
    graph = helper.make_graph(
        nodes=[SlidingWindow_node],
        name="slidingwindow_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="slidingwindow-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_layout("inp", ["N", "H", "W", "C"])

    return model

def prepare_inputs(input_tensor):
    return {"inp": input_tensor}

headers = ["FPGA", "ifm_dim", "ifm_ch", "sf", "simd", "k", "stride", "dw", "idt", "ram_style", "Resources from:",
            "LUT", "LUTRAM", "FF", "SRL", "DSP", "BRAM", "BRAM_18K", "BRAM_36K", "Total_BRAM_18K", "BRAM_efficiency", "URAM", 
            "URAM_efficiency", "Carry", "TargetClockPeriod", "EstimatedClockPeriod", "Delay", "TargetClockFrequency [MHz]",
            "EstimatedClockFrequency [MHz]", "finn_commit", "vivado_version", "vivado_build_no"]
config_headers = ["FPGA", "ifm_dim", "ifm_ch", "sf", "simd", "k", "stride", "dw", "idt", "ram_style", "Resources from:", "finn_commit", "vivado_version", "vivado_build_no"]

def upload_data_to_swu_dashboard(test_parameters, resources):
    
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
        if resources[0] == "hls" or resources[0] == "estimate":
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

# input datatype
@pytest.mark.parametrize("idt", [DataType.BIPOLAR, DataType.INT2, DataType.INT3, DataType.INT4])
# kernel size
@pytest.mark.parametrize("k", [2, 3, 5])
# input dimension
@pytest.mark.parametrize("ifm_dim", [32, 64, 224])
# input channels
@pytest.mark.parametrize("ifm_ch", [3, 32, 128])
# Stride
@pytest.mark.parametrize("stride", [1, 2])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 1, 2, 4])
# depthwise
@pytest.mark.parametrize("dw", [0, 1])
# ram style
@pytest.mark.parametrize("ram_style", ["auto", "distributed", "block", "ultra"])
# Upload to google spreadsheet
@pytest.mark.parametrize("upload", [True])
# Remove artefacts
@pytest.mark.parametrize("cleanup", [True])
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.resource_estimation
def test_fpgadataflow_slidingwindow(idt, k, ifm_dim, ifm_ch, stride, sf, dw, ram_style, upload, cleanup):

    if sf == -1:
        sf = ifm_ch
    simd = ifm_ch // sf
    assert ifm_ch % sf == 0

    ofm_dim = int(((ifm_dim - k) / stride) + 1)

    x = gen_finn_dt_tensor(idt, (1, ifm_dim, ifm_dim, ifm_ch))
    model = make_single_slidingwindow_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt, dw)
    
    #use this only if worksheet doesn't exist in dashboard 
    #(Avoid using this function or checking everytime if the worksheet exists to avoid reaching the gspread usage limits (number of requests per project/user))
    #create_worksheet_in_resource_dashboard(WORKSHEET_NAME, 10, headers)
    #import pdb; pdb.set_trace()
    
    #Streamlining
    model = model.transform(Streamline())

    #Convert to HLS layers
    model = model.transform(to_hls.InferConvInpGen())

    #Dataflow Partitioning
    model = model.transform(CreateDataflowPartition())

    #get the StreamingDataflowPartition
    sdp_node = model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")

    #save the dataflow partition with a different name for easier access
    dataflow_model = ModelWrapper(dataflow_model_filename)

    dataflow_model = dataflow_model.transform(GiveUniqueNodeNames())

    dataflow_model.save("sliding_window_unit_before_synth.onnx")

    test_parameters = {'FPGA': FPGA, 'idt': idt, 'k': k, 'ifm_dim': ifm_dim, 'ifm_ch': ifm_ch, 'sf': sf, 'simd': simd, 'stride': stride, 'dw': dw, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD}
    finn_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace/finn")
    finn_commit = finn_commit.decode("utf-8").strip()

    #PrepareIP, #HLSSynthIP
    dataflow_model = dataflow_model.transform(PrepareIP(FPGA, TARGET_CLK_PERIOD))
    dataflow_model = dataflow_model.transform(HLSSynthIP())

    #skip out of context synthesis if config_dict already exists in finn-resource-dashboard
    config_dict = {'FPGA': FPGA, 'idt': idt, 'k': k, 'ifm_dim': ifm_dim, 'ifm_ch': ifm_ch, 'sf': sf, 'simd': simd, 'stride': stride, 'dw': dw, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD, 'Resources from:': 'synthesis'}
    
    #TODO have to get this from vivado (tcl script) + add to config_dicts
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
            upload_data_to_swu_dashboard(test_parameters, synthesis_resources)

        dataflow_model.save("swu_model_after_oocsynth.onnx")
    
    #skip getting hls estimates if config_dict already exists in finn-resource-dashboard
    config_dict = {'FPGA': FPGA, 'idt': idt, 'k': k, 'ifm_dim': ifm_dim, 'ifm_ch': ifm_ch, 'sf': sf, 'simd': simd, 'stride': stride, 'dw': dw, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD, 'Resources from:': 'hls'}

    if upload:
        matched, row_index = search_in_resource_dashboard(WORKSHEET_NAME, config_dict)
    else:
        matched = False
        
    if not matched:       
        #get resources estimated by hls
        dataflow_model_hls = dataflow_model.transform(AnnotateResources(mode="hls"))
        dataflow_model_hls.save("swu_model_hls.onnx")

        hls_resources = ["hls"]
        custom_ops_hls = getCustomOp(dataflow_model_hls.graph.node[0])
        hls_resources.append(custom_ops_hls.get_nodeattr("res_hls"))
        hls_resources.append({'finn_commit': finn_commit, 'vivado_version': vivado_version, 'vivado_build_no': vivado_build_no})
        
        #get EstimatedClockPeriod
        timing = get_timing(dataflow_model_hls, "hls")
        hls_resources.append(timing[dataflow_model_hls.graph.node[0].name])
        
        if upload:
            upload_data_to_swu_dashboard(test_parameters, hls_resources)
        
        if cleanup:
            dataflow_model_hls.transform(CleanUp())

    #skip getting finn estimates if config_dict already exists in finn-resource-dashboard
    config_dict = {'FPGA': FPGA, 'idt': idt, 'k': k, 'ifm_dim': ifm_dim, 'ifm_ch': ifm_ch, 'sf': sf, 'simd': simd, 'stride': stride, 'dw': dw, 'ram_style': ram_style, 'TargetClockPeriod': TARGET_CLK_PERIOD, 'Resources from:': 'estimate'}

    if upload:
        matched, row_index = search_in_resource_dashboard(WORKSHEET_NAME, config_dict)
    else:
        matched = False
        
    if not matched:  
        #get estimated resources
        dataflow_model_estimate = dataflow_model.transform(AnnotateResources(mode="estimate"))
        dataflow_model_estimate.save("swu_model_estimate.onnx")

        estimate_resources = ["estimate"]
        custom_ops_estimate = getCustomOp(dataflow_model_estimate.graph.node[0])
        estimate_resources.append(custom_ops_estimate.get_nodeattr("res_estimate"))
        estimate_resources.append({'finn_commit': finn_commit, 'vivado_version': vivado_version, 'vivado_build_no': vivado_build_no})

        if upload:
            upload_data_to_swu_dashboard(test_parameters, estimate_resources)
        if cleanup:
            dataflow_model_estimate.transform(CleanUp())

    if cleanup:
        model.transform(CleanUp())
        dataflow_model.transform(CleanUp())