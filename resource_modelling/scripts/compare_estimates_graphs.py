import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
from finn.core.datatype import DataType
import seaborn as sns

#define report folder
#report_folder_name = "output_mobilenetv1-w4a4_U250"
#report_folder_name = "output_mobilenetv1-w4a4_U250_standalone_thresholds"
#report_folder_name = "output_cnv-w1a1_xilinx_u250_xdma_201830_2"
#report_folder_name = "output_tfc-w2a2_xilinx_u250_xdma_201830_2"
#report_folder_name = "output_tfc-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds"
#report_folder_name = "output_tfc-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2"
report_folder_name = "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds"
#report_folder_name = "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2"

report_folder_path = "../results/%s/report" % report_folder_name

#finn_estimate_file = "%s/estimate_layer_resources_copied.json" % report_folder_path
finn_estimate_file = "%s/estimate_layer_resources.json" % report_folder_path
hls_estimate_file = "%s/estimate_layer_resources_hls.json" % report_folder_path
#hls_estimate_file = "%s/estimate_layer_resources_hls_copied.json" % report_folder_path
svr_estimate_file = "%s/estimate_layer_resources_svr.json" % report_folder_path

#synth_resources_file = "%s/post_synth_resources_copied.xml" % report_folder_path
synth_resources_file = "%s/post_synth_resources.xml" % report_folder_path

#resource = 'LUT'
resources = ['LUT', 'BRAM_18K']
#BRAM_18K

fclayer_params_range = {
    #added 10 only for plotting reasons -extrapolation works
    "mh": [10, 16, 64, 128, 256, 512],
    #added 2304 only for plotting reasons -extrapolation works
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
    #added 3 only for plotting reasons -extrapolation works
    "ifm_dim": [3, 4, 8, 16, 32, 64, 224],
    "ifm_ch": [3, 32, 64, 128, 256],
    "stride": [1, 2],
    "ram_style": ["auto", "distributed", "block", "ultra"]
}

# TODO build these indices based on table headers instead of harcoding
"""
restype_to_ind_default = {
        "LUT": 2,
        "LUTRAM": 4,
        "SRL": 5,
        "FF": 6,
        "BRAM_36K": 7,
        "BRAM_18K": 8,
        "DSP48": 9,
    }
"""
restype_to_ind_default = {
        "LUT": 4,
        "LUTRAM": 6,
        "SRL": 7,
        "FF": 8,
        "BRAM_36K": 9,
        "BRAM_18K": 10,
        "DSP48": 12,
    }

with open(finn_estimate_file, 'r') as file:
    dict_finn = json.load(file)

with open(hls_estimate_file, 'r') as file:
    dict_hls = json.load(file)

with open(svr_estimate_file, 'r') as file:
    dict_svr = json.load(file)

layers = list(dict_hls.keys())
layers_df = pd.DataFrame()

for resource in resources:

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
        layers_df["res_finn_lut"] = res_finn
        layers_df["res_hls_lut"] = res_hls
        layers_df["res_svr_lut"] = res_svr
        layers_df["res_synth_lut"] = res_synth
        layers_df["params_svr"] = params_svr
    else:
        layers_df["layer"] = layers
        layers_df["res_finn_bram"] = res_finn
        layers_df["res_hls_bram"] = res_hls
        layers_df["res_svr_bram"] = res_svr
        layers_df["res_synth_bram"] = res_synth
        layers_df["params_svr"] = params_svr
###
layers_svr = ["StreamingFCLayer_Batch", "Thresholding_Batch", "ConvolutionInputGenerator"]

layers_df_reordered = pd.DataFrame()
index_row = []

for layer in layers_svr:
    index_row.append(layers_df.index[layers_df['layer'].str.contains(layer)].tolist())

index_row = [item for sublist in index_row for item in sublist]
layers_df_reordered = layers_df_reordered.append(layers_df.iloc[index_row], ignore_index=True)
layers_df.drop(layers_df.index[index_row], inplace=True)

###remove the layers with out of range parameters
index_in_range = []
index_out_of_range = []

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

for ind in layers_df_reordered.index:
    if 'StreamingFCLayer_Batch' in layers_df_reordered['layer'][ind]:
        comparison_dict = fclayer_params_range
    elif 'Thresholding_Batch' in layers_df_reordered['layer'][ind]:
        comparison_dict = thresholding_params_range
    elif 'ConvolutionInputGenerator' in layers_df_reordered['layer'][ind]:
        comparison_dict = swu_params_range
    params_dict = layers_df_reordered['params_svr'][ind]
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

layers_df_reordered.drop(layers_df_reordered.index[index_out_of_range], inplace=True)
layers_df_reordered.reset_index(drop=True, inplace=True)
###

index_row = []
df_fc = pd.DataFrame()
index_row.append(layers_df_reordered.index[layers_df_reordered['layer'].str.contains("StreamingFCLayer_Batch")].tolist())
index_row = [item for sublist in index_row for item in sublist]
df_fc = df_fc.append(layers_df_reordered.iloc[index_row], ignore_index=True)
 

index_row = []
df_thresh = pd.DataFrame()
index_row.append(layers_df_reordered.index[layers_df_reordered['layer'].str.contains("Thresholding_Batch")].tolist())
index_row = [item for sublist in index_row for item in sublist]
df_thresh = df_thresh.append(layers_df_reordered.iloc[index_row], ignore_index=True)

index_row = []
df_conv = pd.DataFrame()
index_row.append(layers_df_reordered.index[layers_df_reordered['layer'].str.contains("ConvolutionInputGenerator")].tolist())
index_row = [item for sublist in index_row for item in sublist]
df_conv = df_conv.append(layers_df_reordered.iloc[index_row], ignore_index=True)

df_temp = pd.DataFrame()
df_temp["layer"] = ["StreamingFCLayer_Batch", "Thresholding_Batch", "ConvolutionInputGenerator"]
df_temp["total_finn_lut"] = [0, 0, 0]
df_temp["total_finn_bram"] = [0, 0, 0]
df_temp["total_hls_lut"] = [0, 0, 0]
df_temp["total_hls_bram"] = [0, 0, 0]
df_temp["total_svr_lut"] = [0, 0, 0]
df_temp["total_svr_bram"] = [0, 0, 0]
df_temp["total_synth_lut"] = [0, 0, 0]
df_temp["total_synth_bram"] = [0, 0, 0]

df_temp["total_finn_lut"][0] = sum(df_fc['res_finn_lut'])
df_temp["total_finn_lut"][1] = sum(df_thresh['res_finn_lut'])
df_temp["total_finn_lut"][2] = sum(df_conv['res_finn_lut'])

df_temp["total_finn_bram"][0] = sum(df_fc['res_finn_bram'])
df_temp["total_finn_bram"][1] = sum(df_thresh['res_finn_bram'])
df_temp["total_finn_bram"][2] = sum(df_conv['res_finn_bram'])

df_temp["total_hls_lut"][0] = sum(df_fc['res_hls_lut'])
df_temp["total_hls_lut"][1] = sum(df_thresh['res_hls_lut'])
df_temp["total_hls_lut"][2] = sum(df_conv['res_hls_lut'])

df_temp["total_hls_bram"][0] = sum(df_fc['res_hls_bram'])
df_temp["total_hls_bram"][1] = sum(df_thresh['res_hls_bram'])
df_temp["total_hls_bram"][2] = sum(df_conv['res_hls_bram'])

df_temp["total_svr_lut"][0] = sum(df_fc['res_svr_lut'])
df_temp["total_svr_lut"][1] = sum(df_thresh['res_svr_lut'])
df_temp["total_svr_lut"][2] = sum(df_conv['res_svr_lut'])

df_temp["total_svr_bram"][0] = sum(df_fc['res_svr_bram'])
df_temp["total_svr_bram"][1] = sum(df_thresh['res_svr_bram'])
df_temp["total_svr_bram"][2] = sum(df_conv['res_svr_bram'])

df_temp["total_synth_lut"][0] = sum(df_fc['res_synth_lut'])
df_temp["total_synth_lut"][1] = sum(df_thresh['res_synth_lut'])
df_temp["total_synth_lut"][2] = sum(df_conv['res_synth_lut'])

df_temp["total_synth_bram"][0] = sum(df_fc['res_synth_bram'])
df_temp["total_synth_bram"][1] = sum(df_thresh['res_synth_bram'])
df_temp["total_synth_bram"][2] = sum(df_conv['res_synth_bram'])


#Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
df_temp["total_synth_lut_denom"] = np.asarray([(abs(x) + 1) if x == 0 else x for x in df_temp["total_synth_lut"]])
df_temp["total_synth_bram_denom"] = np.asarray([(abs(x) + 1) if x == 0 else x for x in df_temp["total_synth_bram"]])

df_temp["rel_error_finn_lut"] = (abs(df_temp["total_finn_lut"] - df_temp["total_synth_lut"])/df_temp["total_synth_lut_denom"]) * 100
df_temp["rel_error_finn_bram"] = (abs(df_temp["total_finn_bram"] - df_temp["total_synth_bram"])/df_temp["total_synth_bram_denom"]) * 100
#df_temp["rel_error_finn_lut"] = df_temp["total_finn_lut"]/df_temp["total_synth_lut_denom"]
#df_temp["rel_error_finn_bram"] = df_temp["total_finn_bram"]/df_temp["total_synth_bram_denom"]


df_temp["rel_error_hls_lut"] = (abs(df_temp["total_hls_lut"] - df_temp["total_synth_lut"])/df_temp["total_synth_lut_denom"]) * 100
df_temp["rel_error_hls_bram"] = (abs(df_temp["total_hls_bram"] - df_temp["total_synth_bram"])/df_temp["total_synth_bram_denom"]) * 100
#df_temp["rel_error_hls_lut"] = df_temp["total_hls_lut"]/df_temp["total_synth_lut_denom"]
#df_temp["rel_error_hls_bram"] = df_temp["total_hls_bram"]/df_temp["total_synth_bram_denom"]

df_temp["rel_error_svr_lut"] = (abs(df_temp["total_svr_lut"] - df_temp["total_synth_lut"])/df_temp["total_synth_lut_denom"]) * 100
df_temp["rel_error_svr_bram"] = (abs(df_temp["total_svr_bram"] - df_temp["total_synth_bram"])/df_temp["total_synth_bram_denom"]) * 100
#df_temp["rel_error_svr_lut"] = df_temp["total_svr_lut"]/df_temp["total_synth_lut_denom"]
#df_temp["rel_error_svr_bram"] = df_temp["total_svr_bram"]/df_temp["total_synth_bram_denom"]

df_final = pd.DataFrame()
df_final["layer"] = ["FCLayer", "FCLayer", "FCLayer", "FCLayer", "FCLayer", "FCLayer", 
                    "Thresholding", "Thresholding", "Thresholding", "Thresholding", "Thresholding", "Thresholding",
                    "Convolutional",  "Convolutional",  "Convolutional",  "Convolutional",  "Convolutional",  "Convolutional"
                    ] 

df_final["method"] = ["HLS LUT", "FINN LUT", "SVR LUT", "HLS BRAM", "FINN BRAM", "SVR BRAM",
                    "HLS LUT", "FINN LUT", "SVR LUT", "HLS BRAM", "FINN BRAM", "SVR BRAM",
                    "HLS LUT", "FINN LUT", "SVR LUT", "HLS BRAM", "FINN BRAM", "SVR BRAM"
                    ]

df_final["relative error %"] = [df_temp["rel_error_hls_lut"][0], df_temp["rel_error_finn_lut"][0], df_temp["rel_error_svr_lut"][0], 
                    df_temp["rel_error_hls_bram"][0], df_temp["rel_error_finn_bram"][0], df_temp["rel_error_svr_bram"][0],
                    df_temp["rel_error_hls_lut"][1], df_temp["rel_error_finn_lut"][1], df_temp["rel_error_svr_lut"][1], 
                    df_temp["rel_error_hls_bram"][1], df_temp["rel_error_finn_bram"][1], df_temp["rel_error_svr_bram"][1],
                    df_temp["rel_error_hls_lut"][2], df_temp["rel_error_finn_lut"][2], df_temp["rel_error_svr_lut"][2], 
                    df_temp["rel_error_hls_bram"][2], df_temp["rel_error_finn_bram"][2], df_temp["rel_error_svr_bram"][2]
                    ]

#df_final["rel_error_finn_lut"] = df_temp["rel_error_finn_lut"]
#df_final["rel_error_finn_bram"] = df_temp["rel_error_finn_bram"]

#df_final["rel_error_hls_lut"] = df_temp["rel_error_hls_lut"]
#df_final["rel_error_hls_bram"] = df_temp["rel_error_hls_bram"]

#df_final["rel_error_svr_lut"] = df_temp["rel_error_svr_lut"]
#df_final["rel_error_svr_bram"] = df_temp["rel_error_svr_bram"]

#import pdb; pdb.set_trace()

fig = plt.figure(figsize=(20, 11))
#boxplot = df_final.boxplot(showfliers=False, patch_artist=True)
"""
# create another grouped boxplot 
boxplot = sns.boxplot(x = df_final['layer'],
            y = df_final['error'],
            hue = df_final['method'],
            palette = 'Set2')
"""
barplot = sns.barplot(x=df_final['layer'], y=df_final['relative error %'], hue=df_final['method'], palette="flare")
#boxplot.set_title("Relative error of HLS and FINN analytical LUT  estimates for Fully Connected, Thresholding and Convolutional Layers")
#boxplot.set_ylabel('relative error [%]')
fig.savefig('../graphs/plot_box_rel_error_cnv_w2a2_new.png', bbox_inches='tight')
