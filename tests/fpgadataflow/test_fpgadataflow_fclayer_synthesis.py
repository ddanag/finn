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
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.general.multithreshold import multithreshold
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import calculate_signed_dot_prod_range, gen_finn_dt_tensor
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.streamline import Streamline
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources

from finn.analysis.fpgadataflow.get_timing import get_timing
import csv

filename  = "fclayer_resources.csv"
FPGA = "xc7z020clg400-1"
BOARD = "Pynq-Z1"
TARGET_CLK_PERIOD = 5

def make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None):
    mw = W.shape[0]
    mh = W.shape[1]
    assert mh % pe == 0
    assert mw % simd == 0

    # there are two ways to implement bipolar weights and inputs for
    # StreamingFC:
    # - specify their datatypes as such
    # - specify their datatypes as BINARY as use binaryXnorMode
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # we'll internally convert weights/inputs to binary and specify the
        # datatypes as such, and also set the binaryXnorMode attribute to 1
        export_wdt = DataType.BINARY
        export_idt = DataType.BINARY
        binary_xnor_mode = 1
    else:
        export_wdt = wdt
        export_idt = idt
        binary_xnor_mode = 0

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType.BIPOLAR:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    if binary_xnor_mode:
        # convert bipolar to binary
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}

def ram_efficiency_hls(ram_type, ram_hls, wdt, mw, mh):
    bitwidth = wdt.bitwidth()

    if ram_hls == 0:
        return 1

    elif ram_type == "BRAM":    
        return (bitwidth * mw * mh)/(ram_hls * 36 * 512) #wbits/bram_capacity

    elif ram_type == "URAM":
        return (bitwidth * mw * mh)/(ram_hls * 72 * 4096) #wbits/uram_capacity

headers =   ["FPGA", "mh", "mw", "nf", "sf", "pe", "simd", "act", "wdt", "idt", "mem_mode", "Resources from:",
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
    if resources[0] == "hls" or  resources[0] == "synthesis":
        row[headers.index("BRAM_efficiency")] = ram_efficiency_hls("BRAM",  int(row[headers.index("Total_BRAM_18K")]), test_parameters["wdt"], test_parameters["mw"], test_parameters["mh"])
        if int(row[headers.index("URAM")]) != -1:
            row[headers.index("URAM_efficiency")] = ram_efficiency_hls("URAM",  int(row[headers.index("URAM")]), test_parameters["wdt"], test_parameters["mw"], test_parameters["mh"])

    #compute clock freq
    row[headers.index("TargetClockFrequency [MHz]")] = 1/float(row[headers.index("TargetClockPeriod")]) * 1000
    if row[headers.index("EstimatedClockPeriod")] != -1:
        row[headers.index("EstimatedClockFrequency [MHz]")] = 1/float(row[headers.index("EstimatedClockPeriod")]) * 1000

    with open(filename, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

# mem_mode: const or decoupled
@pytest.mark.parametrize("mem_mode", ["const", "decoupled", "external"])
# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType.BIPOLAR, DataType.INT2, DataType.INT4])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType.BIPOLAR, DataType.INT2, DataType.INT4])
# input datatype
@pytest.mark.parametrize("idt", [DataType.BIPOLAR, DataType.INT2, DataType.INT4])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 2, 1])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [16, 64, 128])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [16, 64, 128])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fclayer_synthesis(mem_mode, idt, wdt, act, nf, sf, mw, mh):
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # generate input data
    x = gen_finn_dt_tensor(idt, (1, mw))
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            odt = DataType.UINT32
        else:
            odt = DataType.INT32
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            tdt = DataType.UINT32
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType.INT32

    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)

    #Streamlining
    model = model.transform(Streamline())

    #Convert to HLS layers
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))

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

    dataflow_model.save("my_fclayer_before_synth.onnx")

    test_parameters = {'FPGA': FPGA, 'mh': mh, 'mw': mw, 'nf': nf, 'sf': sf, 'pe': pe, 'simd': simd, 'act': act, 'wdt': wdt, 'idt': idt, 'mem_mode': mem_mode, 'TargetClockPeriod': TARGET_CLK_PERIOD}
    
    #PrepareIP, HLSSynthIP
    dataflow_model = dataflow_model.transform(PrepareIP(FPGA, TARGET_CLK_PERIOD)) #fpgapart - zynq-7000, 5ns clk
    dataflow_model = dataflow_model.transform(HLSSynthIP())

    #get estimated resources
    dataflow_model_estimate = dataflow_model.transform(AnnotateResources(mode="estimate"))
    dataflow_model_estimate.save("fclayer_model_estimate.onnx")

    estimate_resources = ["estimate"]
    custom_ops_estimate = getCustomOp(dataflow_model_estimate.graph.node[1])
    estimate_resources.append(custom_ops_estimate.get_nodeattr("res_estimate"))
    add_to_csv_file(test_parameters, estimate_resources)

    #get resources estimated by hls
    dataflow_model_hls = dataflow_model.transform(AnnotateResources(mode="hls"))
    dataflow_model_hls.save("fclayer_model_hls.onnx")

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
    dataflow_model_synth.save("parent_fclayer_model_synth.onnx")

    custom_ops_synth = getCustomOp(dataflow_model_synth.graph.node[1])
    path_fclayer_model_synth = custom_ops_synth.get_nodeattr("model")
    child_fclayer_model_synth = ModelWrapper(path_fclayer_model_synth)
    child_fclayer_model_synth.save("child_fclayer_model_synth.onnx")

    synthesis_resources = ["synthesis"]
    custom_ops_synth_child = getCustomOp(child_fclayer_model_synth.graph.node[1])
    synthesis_resources.append(custom_ops_synth_child.get_nodeattr("res_synth"))

    #get delay
    timing = get_timing(dataflow_model_synth, "synth")
    synthesis_resources.append(timing)
    add_to_csv_file(test_parameters, synthesis_resources)