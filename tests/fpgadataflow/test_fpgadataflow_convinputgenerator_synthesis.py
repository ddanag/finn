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
import csv

from finn.util.onnx import nchw_to_nhwc

filename  = "slidingwindow_resources.csv"
FPGA = "xc7z020clg400-1"
BOARD = "Pynq-Z1"
TARGET_CLK_PERIOD = 5

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
    ###
    model.set_tensor_layout("inp", ["N", "H", "W", "C"])
    #nchw_to_nhwc(inp, model, idt, reverse=False)

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}

def ram_efficiency_hls(ram_type, ram_hls, wdt, mw, mh):
    bitwidth = wdt.bitwidth()

    if ram_hls == 0:
        return 1

    elif ram_type == "BRAM":    
        return (bitwidth * mw * mh)/(ram_hls * 36 * 512) #wbits/bram_capacity

    elif ram_type == "URAM":
        return (bitwidth * mw * mh)/(ram_hls * 72 * 4096) #wbits/uram_capacity

headers =   ["FPGA", "k", "ifm_ch", "ifm_dim", "ofm_dim", "simd", "stride", "idt", "dw", "Resources from:",
             "LUT", "LUTRAMs", "FF", "SRL", "DSP", "BRAM_18K", "BRAM_36K", "Total_BRAM_18K", "BRAM_efficiency", "URAM", 
             "URAM_efficiency", "Carry", "TargetClockPeriod", "EstimatedClockPeriod", "Delay", "TargetClockFrequency [MHz]",
             "EstimatedClockFrequency [MHz]"]
             
def csv_file_init():
    global headers
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

counter = 0
def add_to_csv_file(test_parameters, resources):
    global counter, headers
    if counter == 0:
        csv_file_init()
        counter = 1

    row = [-1] * len(headers)
    row[headers.index("Total_BRAM_18K")] = 0
    row[headers.index("DSP")] = 0

    for header in headers:
        for key in test_parameters.keys():
            if header == key:
                row[headers.index(header)] = test_parameters[key]
        
        if header == "Resources from:":
            row[headers.index(header)] = resources[0]
        
        res = eval(resources[1])
        for key in res.keys(): 
            if header == key and header != "DSP":
                row[headers.index(header)] = res[key]

                if key == "BRAM_18K":
                    row[headers.index("Total_BRAM_18K")] += int(res[key])
                elif key == "BRAM_36K":
                    row[headers.index("Total_BRAM_18K")] += 2* int(res[key])

            elif "DSP" in key and "DSP" in header:
                row[headers.index("DSP")] += int(res[key])

        if len(resources) > 2:
            timing = resources[2]
            for key in timing.keys(): 
                if header == key:
                    row[headers.index(header)] = timing[key]
                    if resources[0] == "synthesis":
                        row[headers.index("EstimatedClockPeriod")] = row[headers.index("TargetClockPeriod")] - float(timing[key])
                    else:
                        row[headers.index("Delay")] = row[headers.index("TargetClockPeriod")] - float(timing[key])
    
    #add BRAM & URAM efficiencies for hls and synth
    #if resources[0] == "hls" or  resources[0] == "synthesis":
        #row[headers.index("BRAM_efficiency")] = ram_efficiency_hls("BRAM",  int(row[headers.index("Total_BRAM_18K")]), test_parameters["wdt"], test_parameters["mw"], test_parameters["mh"])
        #if int(row[headers.index("URAM")]) != -1:
            #row[headers.index("URAM_efficiency")] = ram_efficiency_hls("URAM",  int(row[headers.index("URAM")]), test_parameters["wdt"], test_parameters["mw"], test_parameters["mh"])

    #compute clock freq
    row[headers.index("TargetClockFrequency [MHz]")] = 1/float(row[headers.index("TargetClockPeriod")]) * 1000
    if row[headers.index("EstimatedClockPeriod")] != -1:
        row[headers.index("EstimatedClockFrequency [MHz]")] = 1/float(row[headers.index("EstimatedClockPeriod")]) * 1000

    with open(filename, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

# input datatype
@pytest.mark.parametrize("idt", [DataType.BIPOLAR, DataType.INT2, DataType.INT4])
# kernel size
@pytest.mark.parametrize("k", [2, 3, 5])
# input dimension
@pytest.mark.parametrize("ifm_dim", [32, 64, 224])
# input channels
@pytest.mark.parametrize("ifm_ch", [8, 16, 32])
# Stride
@pytest.mark.parametrize("stride", [1, 2])
# execution mode
#@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# input channel parallelism ("SIMD")
#@pytest.mark.parametrize("simd", [1, 2])

# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 2, 1])

# depthwise
@pytest.mark.parametrize("dw", [0, 1])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_slidingwindow(
    idt, k, ifm_dim, ifm_ch, stride, sf, dw
):

    if sf == -1:
        sf = ifm_ch
    simd = ifm_ch // sf
    assert ifm_ch % sf == 0

    ofm_dim = int(((ifm_dim - k) / stride) + 1)

    x = gen_finn_dt_tensor(idt, (1, ifm_dim, ifm_dim, ifm_ch))
    model = make_single_slidingwindow_modelwrapper(
        k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt, dw
    )

    #Streamlining
    model = model.transform(Streamline())

    model = model.transform(InferDataLayouts())

    #Convert to HLS layers
    model = model.transform(to_hls.InferConvInpGen())

    #Dataflow Partitioning
    model = model.transform(CreateDataflowPartition())

    #get the StreamingDataflowPartition
    sdp_node = model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")

    # save the dataflow partition with a different name for easier access
    dataflow_model = ModelWrapper(dataflow_model_filename)

    #Insert FIFO
    dataflow_model = dataflow_model.transform(InsertFIFO())
    dataflow_model = dataflow_model.transform(GiveUniqueNodeNames())

    dataflow_model.save("my_slidingwindow_before_synth.onnx")
    
    test_parameters = {'FPGA': FPGA, 'k': k, 'ifm_ch': ifm_ch, 'ifm_dim': ifm_dim, 'ofm_dim': ofm_dim, 'simd': simd, 'stride': stride, 'idt': idt, 'dw': dw, 'TargetClockPeriod': TARGET_CLK_PERIOD}
    
    #PrepareIP, HLSSynthIP, CreateStitchIP
    dataflow_model = dataflow_model.transform(PrepareIP(FPGA, TARGET_CLK_PERIOD)) #fpgapart - zynq-7000, 5ns clk
    dataflow_model = dataflow_model.transform(HLSSynthIP())

    #get estimated resources
    dataflow_model_estimate = dataflow_model.transform(AnnotateResources(mode="estimate"))
    dataflow_model_estimate.save("slidingwindow_model_estimate.onnx")

    estimate_resources = ["estimate"]
    custom_ops_estimate = getCustomOp(dataflow_model_estimate.graph.node[1])
    estimate_resources.append(custom_ops_estimate.get_nodeattr("res_estimate"))
    add_to_csv_file(test_parameters, estimate_resources)

    #get resources estimated by hls
    dataflow_model_hls = dataflow_model.transform(AnnotateResources(mode="hls"))
    dataflow_model_hls.save("slidingwindow_model_hls.onnx")

    hls_resources = ["hls"]
    custom_ops_hls = getCustomOp(dataflow_model_hls.graph.node[1])
    hls_resources.append(custom_ops_hls.get_nodeattr("res_hls"))

    #get EstimatedClockPeriod
    timing = get_timing(dataflow_model_hls, "hls")
    hls_resources.append(timing[dataflow_model_hls.graph.node[1].name])
    add_to_csv_file(test_parameters, hls_resources)
  
    #CreateStitchedIP, Build
    dataflow_model = dataflow_model.transform(CreateStitchedIP(FPGA, TARGET_CLK_PERIOD))
    dataflow_model = dataflow_model.transform(ZynqBuild(platform = BOARD, period_ns = TARGET_CLK_PERIOD))

    #get resouces after synthesis
    dataflow_model_synth = dataflow_model.transform(AnnotateResources(mode="synth"))
    dataflow_model_synth.save("parent_slidingwindow_model_synth.onnx")

    custom_ops_synth = getCustomOp(dataflow_model_synth.graph.node[1])
    path_slidingwindow_model_synth = custom_ops_synth.get_nodeattr("model")
    child_slidingwindow_model_synth = ModelWrapper(path_slidingwindow_model_synth)
    child_slidingwindow_model_synth.save("child_slidingwindow_model_synth.onnx")

    synthesis_resources = ["synthesis"]
    custom_ops_synth_child = getCustomOp(child_slidingwindow_model_synth.graph.node[1])
    synthesis_resources.append(custom_ops_synth_child.get_nodeattr("res_synth"))

    #get delay
    timing = get_timing(dataflow_model_synth, "synth")
    synthesis_resources.append(timing)
    add_to_csv_file(test_parameters, synthesis_resources)

    import pdb; pdb.set_trace()