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

import json
import numpy as np
import math
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import finn.custom_op.registry as registry
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.core.datatype import DataType

def fclayer_res_estimation(inst):

    res_dict = {}

    mh = inst.get_nodeattr("MH")
    mw = inst.get_nodeattr("MW")
    pe = inst.get_nodeattr("PE")
    simd = inst.get_nodeattr("SIMD")
    idt = inst.get_input_datatype().bitwidth()
    wdt = inst.get_weight_datatype().bitwidth()
    mem_mode = inst.get_nodeattr("mem_mode")
    ram_style = inst.get_nodeattr("ram_style")
    no_act = inst.get_nodeattr("noActivation")
    if no_act == 1:
        act = 0
    else:
        act = inst.get_output_datatype().bitwidth()

    dict_input_params ={'mh':mh, 'mw':mw, 'pe':pe, 'simd':simd, 'idt':str(inst.get_input_datatype()), 'idt_strip':idt, 'wdt':wdt, 'mem_mode':mem_mode, 'ram_style':ram_style, 'act':act}

    #resource_classes = ['LUT', 'LUTRAM', 'FF', 'Total_BRAM_18K', 'Carry'] 
    resource_classes = ['LUT', 'FF']
    res_dict["Input_Params"] = dict_input_params

    for res in resource_classes:
        """
        if res == 'LUT':
            if wdt == 1 and idt == 1:
                file_path = '/workspace/finn/resource_modelling/models/FCLayer_%s_model_wdt_idt_bipolar.json' % res
            elif wdt != 1 and idt != 1:
                file_path = '/workspace/finn/resource_modelling/models/FCLayer_%s_model_wdt_idt_not_bipolar.json' % res
            else:
                file_path = '/workspace/finn/resource_modelling/models/FCLayer_%s_model.json' % res
        else:
            file_path = '/workspace/finn/resource_modelling/models/FCLayer_%s_model.json' % res
        """    
        file_path = '/workspace/finn/resource_modelling/models/FCLayer_%s_model.json' % res

        with open(file_path, 'r') as file:
            dict_read = json.load(file)

        estimator_params = dict_read['params']
        X_train_before = np.array(dict_read['X_train_before'])
        X_train = np.array(dict_read['X_train'])
        Y_train = np.array(dict_read['Y_train'])
        
        estimator = SVR()
        estimator = estimator.set_params(**estimator_params)
        estimator.fit(X_train, Y_train)

        #label_encoder
        try:
            label_classes = dict_read['label_classes']
            mem_mode_class = label_classes.index(mem_mode)
        except:
            print("No label classes")
        #TODO add the features list in jsons 
        """       
        if res == 'LUT':
            if wdt == 1 and idt == 1:
                input_set = [[mh, mw, pe, simd, act, mem_mode_class]]
            else:
                input_set = [[mh, mw, pe, simd, wdt, idt, act, mem_mode_class]]
        else:
        """
        if res == 'LUT': 
            input_set = [[mh, mw, pe, simd, wdt, idt]]
        else:
            input_set = [[mh, mw, pe, simd, wdt, idt, act, mem_mode_class]]
            
        feature_scaler = StandardScaler().fit(X_train_before)
        input_set = feature_scaler.transform(input_set)
        
        if dict_read['target_scaler'] == 0:
            svr_estimate = np.exp(estimator.predict(input_set.tolist()))
        elif dict_read['target_scaler'] == 1:
            svr_estimate = estimator.predict(input_set.tolist())
            #TODO add FINN estimate
        else:
            svr_estimate = estimator.predict(input_set.tolist())

        res_dict[res] = svr_estimate.tolist()[0]

    return res_dict

def thresholding_res_estimation(inst):
    
    res_dict = {}

    ich = inst.get_nodeattr("NumChannels")
    pe = inst.get_nodeattr("PE")
    idt = inst.get_nodeattr("inputDataType")
    idt = inst.get_input_datatype().bitwidth()
    numSteps = inst.get_nodeattr("numSteps")

    if numSteps + 1 == 2:
        act = DataType.BIPOLAR
    elif numSteps + 1 == 3:
        act = DataType.TERNARY
    elif numSteps + 1 == abs(DataType.INT2.min()) + abs(DataType.INT2.max()) + 1:
        act = DataType.INT2
    elif numSteps + 1 == abs(DataType.INT3.min()) + abs(DataType.INT3.max()) + 1:
        act = DataType.INT3
    elif numSteps + 1 == abs(DataType.INT4.min()) + abs(DataType.INT4.max()) + 1:
        act = DataType.INT4
    elif numSteps + 1 == abs(DataType.INT5.min()) + abs(DataType.INT5.max()) + 1:
        act = DataType.INT5
    elif numSteps + 1 == abs(DataType.INT6.min()) + abs(DataType.INT6.max()) + 1:
        act = DataType.INT6
    elif numSteps + 1 == abs(DataType.INT7.min()) + abs(DataType.INT7.max()) + 1:
        act = DataType.INT7
    elif numSteps + 1 == abs(DataType.INT8.min()) + abs(DataType.INT8.max()) + 1:
        act = DataType.INT8

    act = act.bitwidth()
    
    mem_mode = inst.get_nodeattr("mem_mode")
    ram_style = inst.get_nodeattr("ram_style")
    
    dict_input_params = {'ich':ich, 'pe':pe, 'idt_strip':idt, 'idt':str(inst.get_input_datatype()), 'numSteps':numSteps, 'act':act, 'mem_mode':mem_mode, 'ram_style':ram_style}
    res_dict["Input_Params"] = dict_input_params
    
    #resource_classes = ['LUT', 'LUTRAM', 'FF', 'Total_BRAM_18K', 'Carry']
    resource_classes = ['LUT', 'FF']

    for res in resource_classes:
        if res == "Total_BRAM_18K" and ram_style == "distributed":
            res_dict[res] = 0
        elif res == "LUTRAM" and ram_style == "block":
            res_dict[res] = 0
        else:
            if res == "Total_BRAM_18K" and ram_style == "block":
                file_path = '/workspace/finn/resource_modelling/models/Thresholding_%s_model_ram_style_block.json' % res
            elif res == "LUTRAM" and ram_style == "distributed":
                file_path = '/workspace/finn/resource_modelling/models/Thresholding_%s_model_ram_style_distributed.json' % res
            else:
                file_path = '/workspace/finn/resource_modelling/models/Thresholding_%s_model.json' % res
            
            with open(file_path, 'r') as file:
                dict_read = json.load(file)

            estimator_params = dict_read['params']
            X_train_before = np.array(dict_read['X_train_before'])
            X_train = np.array(dict_read['X_train'])
            Y_train = np.array(dict_read['Y_train'])

            try:
                label_classes = dict_read['label_classes']
            except:
                print("There are no label classes.")

            estimator = SVR()
            estimator = estimator.set_params(**estimator_params)
            estimator.fit(X_train, Y_train)

            #TODO get both label encoders from json
            if mem_mode == "const":
                mem_mode_class = 0
            elif mem_mode == "decoupled":
                mem_mode_class = 1
            
            #TODO add the features list in jsons 
            if res == "Total_BRAM_18K" or res == "LUTRAM" or res == "LUT":
                input_set = [[ich, pe, idt, act]]
            else:
                ram_style_class = label_classes.index(ram_style)
                input_set = [[ich, pe, idt, act, mem_mode_class, ram_style_class]]

            feature_scaler = StandardScaler().fit(X_train_before)
            input_set = feature_scaler.transform(input_set)
            
            if dict_read['target_scaler'] == 0:
                svr_estimate = np.exp(estimator.predict(input_set.tolist()))
            elif dict_read['target_scaler'] == 1:
                svr_estimate = estimator.predict(input_set.tolist())
                #TODO add FINN estimate
            else:
                svr_estimate = estimator.predict(input_set.tolist())
            
            res_dict[res] = svr_estimate.tolist()[0]

    return res_dict

def convolutioninputgenerator_res_estimation(inst):

    res_dict = {}

    ifm_dim = inst.get_nodeattr("IFMDim")[0]
    ifm_ch = inst.get_nodeattr("IFMChannels")
    simd = inst.get_nodeattr("SIMD")
    k = inst.get_nodeattr("ConvKernelDim")[0]
    stride = inst.get_nodeattr("Stride")[0]
    idt = inst.get_nodeattr("inputDataType")
    idt = inst.get_input_datatype().bitwidth()
    dw = inst.get_nodeattr("depthwise")
    ram_style = inst.get_nodeattr("ram_style")

    dict_input_params = {'ifm_dim':ifm_dim, 'ifm_ch':ifm_ch, 'simd':simd, 'k':k, 'stride':stride, 'idt_strip':idt, 'idt': str(inst.get_input_datatype()), 'dw':dw, 'ram_style':ram_style}
    res_dict["Input_Params"] = dict_input_params

    #resource_classes = ['LUT', 'LUTRAM', 'FF', 'Total_BRAM_18K', 'URAM', 'Carry']
    resource_classes = ['LUT', 'FF']

    for res in resource_classes:
        if res == "Total_BRAM_18K" and (ram_style == "distributed" or ram_style == "ultra"):
            res_dict[res] = 0
        elif res == "LUTRAM" and (ram_style == "block" or ram_style == "ultra"):
            res_dict[res] = 0
        elif res == "URAM" and (ram_style == "block" or ram_style == "distributed"):
            res_dict[res] = 0
        else:
            if res == "Total_BRAM_18K" and ram_style == "block":
                file_path = '/workspace/finn/resource_modelling/models/Sliding_Window_Unit_%s_model_ram_style_block.json' % res
            elif res == "LUTRAM" and ram_style == "distributed":
                file_path = '/workspace/finn/resource_modelling/models/Sliding_Window_Unit_%s_model_ram_style_distributed.json' % res
            elif res == "URAM" and ram_style == "ultra":
                file_path = '/workspace/finn/resource_modelling/models/Sliding_Window_Unit_%s_model_ultra.json' % res
            else:
                file_path = '/workspace/finn/resource_modelling/models/Sliding_Window_Unit_%s_model.json' % res
            with open(file_path, 'r') as file:
                dict_read = json.load(file)

            estimator_params = dict_read['params']
            X_train_before = np.array(dict_read['X_train_before'])
            X_train = np.array(dict_read['X_train'])
            Y_train = np.array(dict_read['Y_train'])
            try:
                label_classes = dict_read['label_classes']
            except:
                print("There are no label classes.")

            estimator = SVR()
            estimator = estimator.set_params(**estimator_params)
            estimator.fit(X_train, Y_train)
            
            #TODO add the features list in jsons 
            if res == "Total_BRAM_18K" or res == "LUTRAM" or res == "URAM":
                input_set = [[ifm_dim, ifm_ch, simd, k, stride, idt, dw]]
            elif res == "LUT":
                input_set = [[ifm_dim, ifm_ch, simd, k, stride, idt]]
            else:
                #label_encoder
                ram_style_class = label_classes.index(ram_style)
                input_set = [[ifm_dim, ifm_ch, simd, k, stride, idt, dw, ram_style_class]]
                
            feature_scaler = StandardScaler().fit(X_train_before)
            input_set = feature_scaler.transform(input_set)
            
            if dict_read['target_scaler'] == 0:
                svr_estimate = np.exp(estimator.predict(input_set.tolist()))
            elif dict_read['target_scaler'] == 1:
                svr_estimate = estimator.predict(input_set.tolist())
                #TODO add FINN estimate
            else:
                svr_estimate = estimator.predict(input_set.tolist())

            res_dict[res] = svr_estimate.tolist()[0]

    return res_dict

def res_estimation_svr(model):
    """Estimates the resources needed for the given model using Support Vector Regression models.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : resource estimation}."""

    res_dict = {}
    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            inst = registry.getCustomOp(node)
            if node.op_type == "StreamingFCLayer_Batch":
                res_dict[node.name] = fclayer_res_estimation(inst)
            elif node.op_type == "Thresholding_Batch":
                res_dict[node.name] = thresholding_res_estimation(inst)
            elif node.op_type == "ConvolutionInputGenerator":
                res_dict[node.name] = convolutioninputgenerator_res_estimation(inst)
    return res_dict
