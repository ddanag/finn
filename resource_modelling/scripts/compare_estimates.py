import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
from finn.core.datatype import DataType

#define report folder
#report_folder_name = "output_mobilenetv1-w4a4_U250"
#report_folder_name = "output_mobilenetv1-w4a4_U250_standalone_thresholds"
#report_folder_name = "output_cnv-w1a1_xilinx_u250_xdma_201830_2"
#report_folder_name = "output_tfc-w2a2_xilinx_u250_xdma_201830_2"
#report_folder_name = "output_tfc-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds"
report_folder_name = "output_tfc-w1a1_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2"
#report_folder_name = "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds"
#report_folder_name = "output_cnv-w2a2_xilinx_u250_xdma_201830_2_standalone_thresholds_folding_x2"

report_folder_path = "../results/%s/report" % report_folder_name

#finn_estimate_file = "%s/estimate_layer_resources_copied.json" % report_folder_path
finn_estimate_file = "%s/estimate_layer_resources.json" % report_folder_path
hls_estimate_file = "%s/estimate_layer_resources_hls.json" % report_folder_path
#hls_estimate_file = "%s/estimate_layer_resources_hls_copied.json" % report_folder_path
svr_estimate_file = "%s/estimate_layer_resources_svr.json" % report_folder_path

#synth_resources_file = "%s/post_synth_resources_copied.xml" % report_folder_path
synth_resources_file = "%s/post_synth_resources.xml" % report_folder_path

resource = 'LUT'
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

###
layers_df = pd.DataFrame()
layers_df["layer"] = layers
layers_df["res_finn"] = res_finn
layers_df["res_hls"] = res_hls
layers_df["res_svr"] = res_svr
layers_df["res_synth"] = res_synth
layers_df["params_svr"] = params_svr

layers_svr = ["StreamingFCLayer_Batch", "Thresholding_Batch", "ConvolutionInputGenerator"]

layers_df_reordered = pd.DataFrame()
index_row = []

for layer in layers_svr:
    index_row.append(layers_df.index[layers_df['layer'].str.contains(layer)].tolist())

index_row = [item for sublist in index_row for item in sublist]
layers_df_reordered = layers_df_reordered.append(layers_df.iloc[index_row], ignore_index=True)
layers_df.drop(layers_df.index[index_row], inplace=True)

###separate layers where params are in range from the ones where params are out of range
index_in_range = []
index_out_of_range = []
layers_df_reordered_in_range = pd.DataFrame()
layers_df_reordered_out_of_range = pd.DataFrame()

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
            #import pdb; pdb.set_trace()
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
        
layers_df_reordered_in_range = layers_df_reordered_in_range.append(layers_df_reordered.iloc[index_in_range], ignore_index=True)
layers_df_reordered_out_of_range = layers_df_reordered_out_of_range.append(layers_df_reordered.iloc[index_out_of_range], ignore_index=True)

#compute total resources on layers for which we have SVR estimates
total_finn = sum(layers_df_reordered['res_finn'])
total_hls = sum(layers_df_reordered['res_hls'])
total_svr = sum(layers_df_reordered['res_svr'])
total_synth = sum(layers_df_reordered['res_synth'])

#
total_finn_in_range = sum(layers_df_reordered_in_range['res_finn'])
total_hls_in_range = sum(layers_df_reordered_in_range['res_hls'])
total_svr_in_range = sum(layers_df_reordered_in_range['res_svr'])
total_synth_in_range = sum(layers_df_reordered_in_range['res_synth'])

###res on thresholding layer in range
temp_df = layers_df_reordered_in_range[layers_df_reordered_in_range['layer'].str.contains('Thresholding_Batch')]
total_finn_thresholding_in_range = sum(temp_df['res_finn'])
total_hls_thresholding_in_range = sum(temp_df['res_hls'])
total_svr_thresholding_in_range = sum(temp_df['res_svr'])
total_synth_thresholding_in_range = sum(temp_df['res_synth'])
###

#import pdb; pdb.set_trace() 
#add the rest of the layers
#layers_df_reordered = layers_df_reordered.append(pd.Series(['Divider', '0', '0', '0', '0'], index = layers_df_reordered.columns), ignore_index=True)
#layers_df_reordered = layers_df_reordered.append(layers_df, ignore_index=True)
final_df = pd.DataFrame()
final_df = final_df.append(layers_df_reordered_in_range, ignore_index=True)
final_df = final_df.append(pd.Series(['Divider', 0, 0, 0, 0, 0], index = final_df.columns), ignore_index=True)
final_df = final_df.append(layers_df_reordered_out_of_range, ignore_index=True)
final_df = final_df.append(pd.Series(['Divider 2', 0, 0, 0, 0, 0], index = final_df.columns), ignore_index=True)
final_df = final_df.append(layers_df, ignore_index=True)
#compute total resources on all layers
total_synth_all = sum(final_df['res_synth'])

fig = plt.figure(figsize=(40, 20))
ax = fig.gca()

width = 1

ax.bar(final_df['layer'], final_df['res_finn'], width, color='g', alpha = 0.5, label='FINN')
ax.bar(final_df['layer'], final_df['res_hls'], 0.8*width, color='b', alpha = 0.5, label='HLS')
ax.bar(final_df['layer'], final_df['res_svr'], 0.6*width, color='purple', alpha = 0.5, label='SVR')
ax.bar(final_df['layer'], final_df['res_synth'], 0.3*width, color='r', label='SYNTH')

###res on fclayer
temp_df = final_df[final_df['layer'].str.contains('StreamingFCLayer_Batch')]
total_finn_fclayer = sum(temp_df['res_finn'])
total_hls_fclayer = sum(temp_df['res_hls'])
total_svr_fclayer = sum(temp_df['res_svr'])
total_synth_fclayer = sum(temp_df['res_synth'])
###

###res on thresholding layer
temp_df = final_df[final_df['layer'].str.contains('Thresholding_Batch')]
total_finn_thresholding = sum(temp_df['res_finn'])
total_hls_thresholding = sum(temp_df['res_hls'])
total_svr_thresholding = sum(temp_df['res_svr'])
total_synth_thresholding = sum(temp_df['res_synth'])
###

###res on convinputgen
temp_df = final_df[final_df['layer'].str.contains('ConvolutionInputGenerator')]
total_finn_swu = sum(temp_df['res_finn'])
total_hls_swu = sum(temp_df['res_hls'])
total_svr_swu = sum(temp_df['res_svr'])
total_synth_swu = sum(temp_df['res_synth'])
###

try:
    x = (total_synth/total_synth_all)*100
except:
    x = 0
textstr = '\n'.join((
        'Total on layers with SVR estimation:',
        ' ',
        'Total FINN = %d' % (total_finn),
        'Total HLS = %d' % (total_hls),
        'Total SVR = %d' % (math.ceil(total_svr)),
        'Total SYNTH = %d' % (total_synth),
        ' ',
        'Total on layers with SVR estimation with param vals "in range":',
        ' ',
        'Total FINN = %d' % (total_finn_in_range),
        'Total HLS = %d' % (total_hls_in_range),
        'Total SVR = %d' % (math.ceil(total_svr_in_range)),
        'Total SYNTH = %d' % (total_synth_in_range),
        ' ',
        'Total on FCLayer:',
        ' ',
        'Total FINN = %d' % (total_finn_fclayer),
        'Total HLS = %d' % (total_hls_fclayer),
        'Total SVR = %d' % (math.ceil(total_svr_fclayer)),
        'Total SYNTH = %d' % (total_synth_fclayer),
        ' ',
        'Total on Thresholding:',
        ' ',
        'Total FINN = %d' % (total_finn_thresholding),
        'Total HLS = %d' % (total_hls_thresholding),
        'Total SVR = %d' % (math.ceil(total_svr_thresholding)),
        'Total SYNTH = %d' % (total_synth_thresholding),
        ' ',
        'Total on Thresholding with param vals "in range":',
        ' ',
        'Total FINN = %d' % (total_finn_thresholding_in_range),
        'Total HLS = %d' % (total_hls_thresholding_in_range),
        'Total SVR = %d' % (math.ceil(total_svr_thresholding_in_range)),
        'Total SYNTH = %d' % (total_synth_thresholding_in_range),
        ' ',
        'Total on ConvInputGenerator:',
        ' ',
        'Total FINN = %d' % (total_finn_swu),
        'Total HLS = %d' % (total_hls_swu),
        'Total SVR = %d' % (math.ceil(total_svr_swu)),
        'Total SYNTH = %d' % (total_synth_swu),
        ' ',
        'Total SYNTH all layers = %d' % (total_synth_all),
        ' ',
        'FCLayer + Thresholding + SWU layers resources',
        'represent %.2f %s from whole NN resources' % (x, '%'),
        ' ',
        ))
plt.subplots_adjust(right=0.8)
# figtext() takes positional arguments x (0.93) and y (0.5) and a string. The bbox=dict(facecolor='white') creates a box around the text with a white facecolor.
side_text = plt.figtext(0.82, 0.5, textstr, size='x-large', bbox=dict(facecolor='white'))

ax.set_ylabel('%s' % resource)
ax.set_xlabel('Layers')
ax.legend()

plt.xticks(rotation=90)

fig.savefig('../results/comparison_plot_%s_%s.png' % (report_folder_name, resource), bbox_inches='tight')