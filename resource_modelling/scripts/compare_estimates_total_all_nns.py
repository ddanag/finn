import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
from finn.core.datatype import DataType
import seaborn as sns

def extract_data_from_files(resources, finn_estimate_previous_file, finn_estimate_file, hls_estimate_file, svr_estimate_file, synth_resources_file):

    restype_to_ind_default = {
            "LUT": 4,
            "LUTRAM": 6,
            "SRL": 7,
            "FF": 8,
            "BRAM_36K": 9,
            "BRAM_18K": 10,
            "DSP48": 12,
        }

    with open(finn_estimate_previous_file, 'r') as file:
        dict_finn_previous = json.load(file)

    with open(finn_estimate_file, 'r') as file:
        dict_finn = json.load(file)

    with open(hls_estimate_file, 'r') as file:
        dict_hls = json.load(file)

    with open(svr_estimate_file, 'r') as file:
        dict_svr = json.load(file)

    layers = list(dict_hls.keys())
    layers_df = pd.DataFrame()

    for resource in resources:
        res_finn_previous = []
        res_finn = []
        res_hls = []
        res_svr = []
        res_synth = []
        params_svr = []

        #post_synth_resources_xml
        tree = ET.parse(synth_resources_file)
        root = tree.getroot()
        all_cells = root.findall(".//tablecell")
        # strip all whitespace from table cell contents
        for cell in all_cells:
            cell.attrib["contents"] = cell.attrib["contents"].strip()

        for layer in layers:
            #row = root.findall(".//*[@contents='StreamingDataflowPartition_1_%s']/.." %layer)
            row = root.findall(".//*[@contents='%s']/.." %layer)
            try:
                if row != []:
                    row = row[0].getchildren()
                    if resource == 'BRAM_18K':
                        ind_18k = restype_to_ind_default[resource]
                        ind_36k = restype_to_ind_default['BRAM_36K']
                        res_synth.append(int(row[ind_18k].attrib["contents"]) + int(row[ind_36k].attrib["contents"])*2)
                    else:
                        ind = restype_to_ind_default[resource]
                        res_synth.append(int(row[ind].attrib["contents"]))
                else:
                    res_synth.append(0)
            except:
                res_synth.append(0) 

        for layer in layers:
            try:
                res_finn_previous.append(dict_finn_previous[layer][resource])
            except:
                res_finn_previous.append(0) 
            try:
                res_finn.append(dict_finn[layer][resource])
            except:
                res_finn.append(0) 
            try:
                res_hls.append(int(dict_hls[layer][resource])) 
            except:
                res_hls.append(0) 
            try:
                if resource == 'BRAM_18K':
                    if dict_svr[layer]['Total_BRAM_18K'] < 0:
                        dict_svr[layer]['Total_BRAM_18K'] = 0
                    res_svr.append(dict_svr[layer]['Total_BRAM_18K'])
                else:
                    if dict_svr[layer][resource] < 0:
                        dict_svr[layer][resource] = 0
                    res_svr.append(dict_svr[layer][resource])
                params_svr.append(dict_svr[layer]['Input_Params'])
            except:
                res_svr.append(0)
                params_svr.append(0)

        if resource == "LUT":
            layers_df["layer"] = layers
            layers_df["res_finn_previous_lut"] = res_finn_previous
            layers_df["res_finn_lut"] = res_finn
            layers_df["res_hls_lut"] = res_hls
            layers_df["res_svr_lut"] = res_svr
            layers_df["res_synth_lut"] = res_synth
            layers_df["params_svr"] = params_svr
        else:
            layers_df["layer"] = layers
            layers_df["res_finn_previous_bram"] = res_finn_previous
            layers_df["res_finn_bram"] = res_finn
            layers_df["res_hls_bram"] = res_hls
            layers_df["res_svr_bram"] = np.ceil(res_svr)
            layers_df["res_synth_bram"] = res_synth
            layers_df["params_svr"] = params_svr


    return layers_df

def datatype_strip(x):
    if "DataType.UINT" in str(x):
        return int(x.strip("DataType.UINT"))
    elif "DataType.INT" in str(x):
        return int(x.strip("DataType.INT"))
    elif "DataType.FLOAT" in str(x):
        return int(x.strip("DataType.FLOAT"))
    elif x in ["DataType.BINARY", "DataType.BIPOLAR"]:
        return 1
    elif x == "DataType.TERNARY":
        return 2
    elif x == "None":
        return 0
    else:
        return x

def reorder_and_remove_unwanted_layers(layers_df):
    #reorder layers and remove those not in ["StreamingFCLayer_Batch", "Thresholding_Batch", "ConvolutionInputGenerator"]
    layers_svr = ["StreamingFCLayer_Batch", "Thresholding_Batch", "ConvolutionInputGenerator"]

    layers_df_reordered = pd.DataFrame()
    index_row = []

    for layer in layers_svr:
        index_row.append(layers_df.index[layers_df['layer'].str.contains(layer)].tolist())

    index_row = [item for sublist in index_row for item in sublist]
    layers_df_reordered = layers_df_reordered.append(layers_df.iloc[index_row], ignore_index=True)
    layers_df.drop(layers_df.index[index_row], inplace=True)

    return layers_df_reordered

def remove_layers_with_out_of_range_parameters(layers_df):

    fclayer_params_range = {
                            #added 10 only for plotting reasons -extrapolation seems to work
                            "mh": [10, 16, 64, 128, 256, 512],
                            "mw": [16, 64, 128, 256, 512, 1024, 2048, 2304],
                            "idt": [DataType.BIPOLAR, DataType.INT2, DataType.INT4],
                            "wdt": [DataType.BIPOLAR, DataType.INT2, DataType.INT4],
                            "act": [None, DataType.BIPOLAR, DataType.INT2, DataType.INT4],
                            "mem_mode": ["const", "decoupled", "external"],
                            }

    thresholding_params_range = {
                                "act": [DataType.BIPOLAR, DataType.INT2, DataType.INT3, DataType.INT4, DataType.INT5],
                                "idt": [DataType.UINT12, DataType.UINT16, DataType.UINT20, DataType.UINT24, DataType.UINT28, DataType.UINT32],
                                "ich": [3, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 512, 784],
                                "mem_mode": ["const", "decoupled"],
                                "ram_style": ["auto", "distributed", "block"]
                                }

    swu_params_range = {
                        "idt": [DataType.BIPOLAR, DataType.INT2, DataType.INT3, DataType.INT4],
                        "k": [2, 3, 5],
                        #added 3 only for plotting reasons -extrapolation seems to work
                        "ifm_dim": [3, 4, 8, 16, 32, 64, 224],
                        "ifm_ch": [3, 32, 64, 128, 256],
                        "stride": [1, 2],
                        "ram_style": ["auto", "distributed", "block", "ultra"]
                        }

    index_in_range = []
    index_out_of_range = []

    for ind in layers_df.index:
        if 'StreamingFCLayer_Batch' in layers_df['layer'][ind]:
            comparison_dict = fclayer_params_range
        elif 'Thresholding_Batch' in layers_df['layer'][ind]:
            comparison_dict = thresholding_params_range
        elif 'ConvolutionInputGenerator' in layers_df['layer'][ind]:
            comparison_dict = swu_params_range
        params_dict = layers_df['params_svr'][ind]
        if params_dict != 0:
            matched = False
            for key in comparison_dict.keys():
                params_dict[key] = datatype_strip(str(params_dict[key]))
                for i, elem in enumerate(comparison_dict[key]):
                    comparison_dict[key][i] = datatype_strip(str(comparison_dict[key][i]))
                if (params_dict[key] in comparison_dict[key]) or (int(params_dict[key]) >= int(comparison_dict[key][0]) and int(params_dict[key]) <= int(comparison_dict[key][-1])):
                    matched = True
                else:
                    matched = False
                    break
            if matched:
                index_in_range.append(ind) 
            else:
                index_out_of_range.append(ind) 

    layers_df.drop(layers_df.index[index_out_of_range], inplace=True)
    layers_df.reset_index(drop=True, inplace=True)
    
    return layers_df

def compute_and_add_svr_finn_hybrid_model_result_lut(totals_df):
    fclayer_params_range = {
                            #added 10 only for plotting reasons -extrapolation seems to work
                            "mh": [10, 16, 64, 128, 256, 512],
                            "mw": [16, 64, 128, 256, 512, 1024, 2048, 2304],
                            "idt": [DataType.BIPOLAR, DataType.INT2, DataType.INT4],
                            "wdt": [DataType.BIPOLAR, DataType.INT2, DataType.INT4],
                            "act": [None, DataType.BIPOLAR, DataType.INT2, DataType.INT4],
                            "mem_mode": ["const", "decoupled", "external"],
                            }

    thresholding_params_range = {
                                "act": [DataType.BIPOLAR, DataType.INT2, DataType.INT3, DataType.INT4, DataType.INT5],
                                "idt": [DataType.UINT12, DataType.UINT16, DataType.UINT20, DataType.UINT24, DataType.UINT28, DataType.UINT32],
                                "ich": [3, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 512, 784],
                                "mem_mode": ["const", "decoupled"],
                                "ram_style": ["auto", "distributed", "block"]
                                }

    swu_params_range = {
                        "idt": [DataType.BIPOLAR, DataType.INT2, DataType.INT3, DataType.INT4],
                        "k": [2, 3, 5],
                        #added 3 only for plotting reasons -extrapolation seems to work
                        "ifm_dim": [3, 4, 8, 16, 32, 64, 224],
                        "ifm_ch": [3, 32, 64, 128, 256],
                        "stride": [1, 2],
                        "ram_style": ["auto", "distributed", "block", "ultra"]
                        }

    res_svr_finn = []
    index_in_range = []
    index_out_of_range = []

    for ind in totals_df.index:
        if 'StreamingFCLayer_Batch' in totals_df['layer'][ind]:
            comparison_dict = fclayer_params_range
        elif 'Thresholding_Batch' in totals_df['layer'][ind]:
            comparison_dict = thresholding_params_range
        elif 'ConvolutionInputGenerator' in totals_df['layer'][ind]:
            comparison_dict = swu_params_range
        params_dict = totals_df['params_svr'][ind]
        if params_dict != 0:
            matched = False
            for key in comparison_dict.keys():
                params_dict[key] = datatype_strip(str(params_dict[key]))
                for i, elem in enumerate(comparison_dict[key]):
                    comparison_dict[key][i] = datatype_strip(str(comparison_dict[key][i]))
                if (params_dict[key] in comparison_dict[key]):
                    matched = True
                else:
                    value = min(comparison_dict[key], key=lambda x:abs(int(x)-int(params_dict[key])))
                    value = int(value)
                    if (int(params_dict[key]) >= value - value/5) and (int(params_dict[key]) <= value + value/5):
                        matched = True    
                    else:
                        matched = False
                        break
            if matched:
                index_in_range.append(ind) 
            else:
                index_out_of_range.append(ind) 

    for ind in totals_df.index:
        if ind in index_in_range:
            res_svr_finn.append(totals_df['res_svr_lut'][ind])
        elif ind in index_out_of_range:
            res_svr_finn.append(totals_df['res_finn_lut'][ind])

    totals_df["res_svr_finn_lut"] = res_svr_finn

    return totals_df

def get_data_from_the_folders(resources, list_of_folders_name):

    totals_df = pd.DataFrame()

    for folder in list_of_folders_name:
        report_folder_path = "../results/%s/report" % folder
        
        finn_estimate_previous_file = "%s/estimate_layer_resources_previous.json" % report_folder_path
        finn_estimate_file = "%s/estimate_layer_resources.json" % report_folder_path
        hls_estimate_file = "%s/estimate_layer_resources_hls.json" % report_folder_path
        svr_estimate_file = "%s/estimate_layer_resources_svr.json" % report_folder_path
        synth_resources_file = "%s/post_synth_resources.xml" % report_folder_path

        layers_df = extract_data_from_files(resources, finn_estimate_previous_file, finn_estimate_file, hls_estimate_file, svr_estimate_file, synth_resources_file)

        layers_df = reorder_and_remove_unwanted_layers(layers_df)
        #layers_df = remove_layers_with_out_of_range_parameters(layers_df)
        if(resources == ['LUT']):
            layers_df = compute_and_add_svr_finn_hybrid_model_result_lut(layers_df)

        layers_df['layer'] = list_of_folders_name.index(folder)/len(layers_df)
        totals_df = totals_df.append(layers_df.sum(), ignore_index=True)

    totals_df = totals_df.rename(columns={"layer": "BNN"})
    return totals_df

def plot_results_on_all_bnns(resource, totals_df):
    
    totals_df['BNN'] = list_of_bnns
    totals_df = totals_df.rename(columns={"res_hls_lut": "HLS", "res_finn_previous_lut": "FINN (original)", "res_finn_lut": "FINN (updated)", "res_svr_lut": "SVR", "res_svr_finn_lut": "FINN + SVR", "res_synth_lut": "SYNTH (ground truth)"})
    totals_df = totals_df.rename(columns={"res_hls_bram": "HLS", "res_finn_previous_bram": "FINN (original)", "res_finn_bram": "FINN (updated)", "res_svr_bram": "SVR", "res_synth_bram": "SYNTH (ground truth)"})
    if resource == 'LUT':
        totals_df = pd.melt(totals_df, id_vars=['BNN'], value_vars=['HLS', 'FINN (original)', 'FINN (updated)', 'SVR', 'FINN + SVR', 'SYNTH (ground truth)'], var_name="Method", value_name=resource)
    else:
        totals_df = pd.melt(totals_df, id_vars=['BNN'], value_vars=['HLS', 'FINN (original)', 'FINN (updated)', 'SVR', 'SYNTH (ground truth)'], var_name="Method", value_name=resource)
    
    fig = plt.figure(figsize=(20, 11))
    barplot = sns.barplot(x='BNN', y=resource, hue='Method', data=totals_df, palette="magma")
    plt.xticks(rotation=45)
    barplot.set_title("Total resources on BNNs [%s]" % resource)
    plt.savefig('../results_on_bnns_%s_short_list.png' % resource, bbox_inches='tight')

def plot_relative_errors_on_totals(resource, totals_df):
    if resource == 'LUT':
        #Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
        totals_df["res_synth_lut_denom"] = np.asarray([(abs(x) + 1) if x == 0 else x for x in totals_df["res_synth_lut"]])

        totals_df["rel_error_finn_lut"] = (abs(totals_df["res_finn_lut"] - totals_df["res_synth_lut"])/totals_df["res_synth_lut_denom"]) * 100
        totals_df["rel_error_finn_previous_lut"] = (abs(totals_df["res_finn_previous_lut"] - totals_df["res_synth_lut"])/totals_df["res_synth_lut_denom"]) * 100
        totals_df["rel_error_hls_lut"] = (abs(totals_df["res_hls_lut"] - totals_df["res_synth_lut"])/totals_df["res_synth_lut_denom"]) * 100
        totals_df["rel_error_svr_lut"] = (abs(totals_df["res_svr_lut"] - totals_df["res_synth_lut"])/totals_df["res_synth_lut_denom"]) * 100
        totals_df["rel_error_svr_finn_lut"] = (abs(totals_df["res_svr_finn_lut"] - totals_df["res_synth_lut"])/totals_df["res_synth_lut_denom"]) * 100

    elif resource == 'BRAM_18K':
        #Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
        totals_df["res_synth_bram_denom"] = np.asarray([(abs(x) + 1) if x == 0 else x for x in totals_df["res_synth_bram"]])

        totals_df["rel_error_finn_bram"] = (abs(totals_df["res_finn_bram"] - totals_df["res_synth_bram"])/totals_df["res_synth_bram_denom"]) * 100
        totals_df["rel_error_finn_previous_bram"] = (abs(totals_df["res_finn_previous_bram"] - totals_df["res_synth_bram"])/totals_df["res_synth_bram_denom"]) * 100
        totals_df["rel_error_hls_bram"] = (abs(totals_df["res_hls_bram"] - totals_df["res_synth_bram"])/totals_df["res_synth_bram_denom"]) * 100
        totals_df["rel_error_svr_bram"] = (abs(totals_df["res_svr_bram"] - totals_df["res_synth_bram"])/totals_df["res_synth_bram_denom"]) * 100

 
    if resource == 'LUT':
        totals_df = totals_df.drop(columns=["res_hls_lut", "res_finn_previous_lut", "res_finn_lut", "res_svr_lut", "res_svr_finn_lut", "res_synth_lut"])
        totals_df = totals_df.rename(columns={"rel_error_hls_lut": "HLS", "rel_error_finn_previous_lut": "FINN (original)", "rel_error_finn_lut": "FINN", "rel_error_svr_lut": "SVR", "rel_error_svr_finn_lut": "FINN + SVR"})
        totals_df = pd.melt(totals_df, id_vars=['BNN'], value_vars=['HLS', 'FINN (original)', 'FINN', 'SVR', 'FINN + SVR'], var_name="Method", value_name=resource)
    else:
        totals_df = totals_df.drop(columns=["res_hls_bram", "res_finn_previous_bram", "res_finn_bram", "res_svr_bram", "res_synth_bram"])
        totals_df = totals_df.rename(columns={"rel_error_hls_bram": "HLS", "rel_error_finn_previous_bram": "FINN (original)", "rel_error_finn_bram": "FINN", "rel_error_svr_bram": "SVR"})
        totals_df = pd.melt(totals_df, id_vars=['BNN'], value_vars=['HLS', 'FINN (original)', 'FINN', 'SVR'], var_name="Method", value_name=resource)
    
    fig = plt.figure(figsize=(20, 11))
    barplot = sns.barplot(x='BNN', y=resource, hue='Method', data=totals_df, palette="magma")
    plt.xticks(rotation=45)
    barplot.set(ylabel='Relative Error [%]')
    barplot.set_title("Total resources relative errors on BNNs [%s]" % resource)
    plt.savefig('../rel_error_on_bnns_%s_short_list.png' % resource, bbox_inches='tight')
    #import pdb; pdb.set_trace()

""" 
list_of_folders_name = ["output_tfc-w1a1_xilinx_u250_xdma_201830_2_clock_period_3ns",
                        "output_tfc-w1a1_xilinx_u250_xdma_201830_2_clock_period_3ns_folding_x2",
                        "output_tfc-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_tfc-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_tfc-w1a1_xilinx_u250_xdma_201830_2_clock_period_10ns",
                        "output_tfc-w1a1_xilinx_u250_xdma_201830_2_clock_period_10ns_folding_x2",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_clock_period_3ns",
                        #"output_tfc-w2a2_xilinx_u250_xdma_201830_2_clock_period_3ns_folding_x2",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_clock_period_10ns",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_clock_period_10ns_folding_x2",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_clock_period_3ns",
                        #"output_cnv-w1a1_xilinx_u250_xdma_201830_2_clock_period_3ns_folding_x2",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_clock_period_10ns",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_clock_period_10ns_folding_x2",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_clock_period_3ns",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_clock_period_3ns_folding_x2",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_clock_period_10ns",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_clock_period_10ns_folding_x2"
                        ]
list_of_bnns = ['tfc-w1a1 [3ns]',
                'tfc-w1a1-fx2 [3ns]',
                'tfc-w1a1 [5ns]',
                'tfc-w1a1-fx2 [5ns]',
                'tfc-w1a1 [10ns]',
                'tfc-w1a1-fx2 [10ns]',
                'tfc-w2a2 [3ns]',
                #'tfc-w2a2-fx2 [3ns]',
                'tfc-w2a2 [5ns]',
                'tfc-w2a2-fx2 [5ns]',
                'tfc-w2a2 [10ns]',
                'tfc-w2a2-fx2 [10ns]',
                'cnv-w1a1 [3ns]',
                #'cnv-w1a1-fx2 [3ns]',
                'cnv-w1a1 [5ns]',
                'cnv-w1a1-fx2 [5ns]',
                'cnv-w1a1 [10ns]',
                'cnv-w1a1-fx2 [10ns]',
                'cnv-w2a2 [3ns]',
                'cnv-w2a2-fx2 [3ns]',
                'cnv-w2a2 [5ns]',
                'cnv-w2a2-fx2 [5ns]',
                'cnv-w2a2 [10ns]',
                'cnv-w2a2-fx2 [10ns]',
                 ]
"""
list_of_folders_name = ["output_tfc-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_tfc-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_tfc-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_cnv-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds",
                        "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2",
                        ]

list_of_bnns = ['tfc-w1a1',
                'tfc-w1a1-fx2',
                'tfc-w2a2',
                'tfc-w2a2-fx2',
                'cnv-w1a1',
                'cnv-w1a1-fx2',
                'cnv-w2a2',
                'cnv-w2a2-fx2',
                 ]

resources = ['LUT', 'BRAM_18K']

for resource in resources:
    totals_df = get_data_from_the_folders([resource], list_of_folders_name)
    plot_results_on_all_bnns(resource, totals_df)
    plot_relative_errors_on_totals(resource, totals_df)
