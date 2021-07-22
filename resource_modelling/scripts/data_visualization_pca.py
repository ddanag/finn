import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
from finn.util.gdrive import *
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph

#define the worksheet name from finn-resource-dashboard
#worksheet_name = "FCLayer_resources"
#worksheet_name = "Thresholding_layer_resources"
worksheet_name = "Sliding_Window_layer_resources"

#define the directory name where to save the graphs
#directory_name = "FCLayer"
#directory_name = "Thresholding"
directory_name = "Sliding_Window_Unit"

#fclayer
#features = ["mh", "mw", "pe", "simd", "wdt", "idt", "act", "mem_mode"]

#thresholding
#features = ["ich", "pe", "idt", "act", "mem_mode", "ram_style"]

#swu 
features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "dw", "idt", "ram_style"]
target = "LUT"

##create the directory
new_dir_path = "../graphs/%s" % directory_name
try:
    os.mkdir(new_dir_path)
except OSError:
    print ("Creation of the directory %s failed" % new_dir_path)
else:
    print ("Successfully created the directory %s " % new_dir_path)

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)

# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)
print(df)

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

#DataType Strip on features
for feature in features:
    df[feature] = df[feature].apply(datatype_strip)

    if feature == 'ram_style' or feature == 'mem_mode':
        label_encoder = LabelEncoder()
        df[feature] = label_encoder.fit_transform(df[feature])

df['LUTRAM'] = df['LUTRAM'].replace('-', 0)

#separate the dataframe in 3 sub-dataframes:
#   - one which contains the estimated resources
#   - one which contains the hls estimated resources
#   - one which contains the resources reported after Vivado synthesis
#df_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
#df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
df_synth = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

# Separating out the features
x = df_synth.loc[:, features].values
#x_hls = df_hls.loc[:, features].values
# Separating out the target
y = df_synth.loc[:,[target]].values
#y_hls = df_hls.loc[:,[target]].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#x_hls = StandardScaler().fit_transform(x_hls)

pca = PCA()
principalComponents = pca.fit_transform(x)

if worksheet_name == "Thresholding_layer_resources":
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'])
else:
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8'])

finalDf = pd.concat([principalDf, df_synth[[target]]], axis = 1)

print(pca.explained_variance_ratio_)

"""
pca_hls = PCA()
principalComponents_hls = pca_hls.fit_transform(x_hls)
if worksheet_name == "Thresholding_layer_resources":
    principalDf_hls = pd.DataFrame(data = principalComponents_hls, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'])
else:
    principalDf_hls = pd.DataFrame(data = principalComponents_hls, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8'])

finalDf_hls = pd.concat([principalDf_hls, df_hls[[target]]], axis = 1)

print(pca_hls.explained_variance_ratio_)
"""
figure, correlation_matrix = plot_pca_correlation_graph(x, 
                                                        features,
                                                        dimensions=(1, 2),
                                                        figure_axis_size=10)
print(correlation_matrix)
figure.savefig('../graphs/%s/pca_correlation_circle.png' % (directory_name), bbox_inches='tight')

"""
fig = plt.figure(figsize=(20, 11))
ax = fig.gca()

ax.scatter(finalDf_hls['pc1'], finalDf_hls['pc2'], marker="o", s=100, facecolors='none', edgecolors='b', label='LUTs, hls')
ax.scatter(finalDf['pc1'], finalDf['pc2'],marker="x", s=50, color='r', label='LUTs, synth')

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
leg = ax.legend()
fig.savefig('../graphs/%s/swu_pca.png' % (directory_name), bbox_inches='tight')
"""