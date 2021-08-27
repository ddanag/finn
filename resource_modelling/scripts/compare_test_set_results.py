import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

def plot_fclayer_models_test_set_results_lut():
    general_without_fu_model_file_name = "test_set_results_FCLayer_LUT_general_without_fully_unfolded_configs"
    general_with_fu_model_file_name = "test_set_results_FCLayer_LUT_general_with_fully_unfolded_configs"
    general_augmentation_model_file_name = "test_set_results_FCLayer_LUT_general_augmentation"
    specialized_model_file_name = "test_set_results_FCLayer_LUT_specialized"
    specialized_augmentation_model_file_name = "test_set_results_FCLayer_LUT_specialized_augmentation"

    general_without_fu_model_folder_path = "../test_set_results/FCLayer/%s.csv" % general_without_fu_model_file_name
    general_with_fu_model_folder_path = "../test_set_results/FCLayer/%s.csv" % general_with_fu_model_file_name
    general_augmentation_model_folder_path = "../test_set_results/FCLayer/%s.csv" % general_augmentation_model_file_name
    specialized_model_folder_path = "../test_set_results/FCLayer/%s.csv" % specialized_model_file_name
    specialized_augmentation_model_folder_path = "../test_set_results/FCLayer/%s.csv" % specialized_augmentation_model_file_name

    df_general_without_fu_model = pd.read_csv(general_without_fu_model_folder_path)
    df_general_with_fu_model = pd.read_csv(general_with_fu_model_folder_path)
    df_general_augmentation_model = pd.read_csv(general_augmentation_model_folder_path)
    df_specialized_model = pd.read_csv(specialized_model_folder_path)
    df_specialized_augmentation_model = pd.read_csv(specialized_augmentation_model_folder_path)

    #import pdb; pdb.set_trace()

    df_plot = pd.DataFrame()
    df_plot['HLS GM (FD)'] = df_general_with_fu_model['hls_rel_error']
    df_plot['FINN GM (FD)'] = df_general_with_fu_model['finn_rel_error']
    df_plot['SVR GM (FD)'] = df_general_with_fu_model['svr_rel_error']

    df_plot['HLS GM (PD)'] = df_general_without_fu_model['hls_rel_error']
    df_plot['FINN GM (PD)'] = df_general_without_fu_model['finn_rel_error']
    df_plot['SVR GM (PD)'] = df_general_without_fu_model['svr_rel_error']

    #recheck if they are the same - hls and finn
    #df_plot['HLS (general model + aug)'] = df_general_augmentation_model['hls_rel_error']
    #df_plot['FINN (general model + aug)'] = df_general_augmentation_model['finn_rel_error']
    df_plot['SVR GM (PD + AUG)'] = df_general_augmentation_model['svr_rel_error']

    df_plot['HLS SM'] = df_specialized_model['hls_rel_error']
    df_plot['FINN SM'] = df_specialized_model['finn_rel_error']
    df_plot['SVR SM'] = df_specialized_model['svr_rel_error']
    #recheck if they are the same - hls and finn
    #df_plot['HLS (specialized model + aug)'] = df_specialized_augmentation_model['hls_rel_error']
    #df_plot['FINN (specialized model + aug)'] = df_specialized_augmentation_model['finn_rel_error']
    df_plot['SVR SM (PD + AUG)'] = df_specialized_augmentation_model['svr_rel_error']

    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=True, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    colors = ['lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightyellow']
    #import pdb; pdb.set_trace()
    
    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation = 45)
    plt.title('FCLayer - LUT estimation model - Test Set Results')
    plt.ylabel('Relative error [%]')
    fig.savefig('../test_set_results/FCLayer/test_set_results_plot_luts_with_fu_with_outliers.png', bbox_inches='tight')

def add_newest_finn_estimation_to_the_csv_file(filename):
    #finn estimate computed with the clasifier
    df_updated_finn_estimate = pd.read_csv("../test_set_results/updated_fclayer_database_finn_estimate.csv")
    filepath = "../test_set_results/FCLayer/%s.csv" % filename
    #the other csv file that needs to be updated
    df_initial = pd.read_csv(filepath)
    df_initial['bram_new_finn_estimate'] = -1
    #remove rows of df_updated_finn_estimate not found in df_initial
    #copy to df_initial finn_new_estimate
    if (filename == 'test_set_results_FCLayer_Total_BRAM_18K_specialized_min_fu'):
        parameters = ['mh', 'mw', 'pe', 'simd', 'wdt', 'idt']
    else:
        parameters = ['mh', 'mw', 'pe', 'simd', 'wdt', 'idt', 'act', 'mem_mode']
    for index1, row1 in df_initial[parameters].iterrows():
        if (filename == 'test_set_results_FCLayer_Total_BRAM_18K_specialized_min_fu'):
            df_temp = df_updated_finn_estimate.loc[(df_updated_finn_estimate.mh == row1['mh']) & (df_updated_finn_estimate.mw == row1['mw']) & (df_updated_finn_estimate.pe == row1['pe']) & (df_updated_finn_estimate.simd == row1['simd']) & (df_updated_finn_estimate.idt == row1['idt']) & (df_updated_finn_estimate.wdt == row1['wdt'])]
        else:
            df_temp = df_updated_finn_estimate.loc[(df_updated_finn_estimate.mh == row1['mh']) & (df_updated_finn_estimate.mw == row1['mw']) & (df_updated_finn_estimate.pe == row1['pe']) & (df_updated_finn_estimate.simd == row1['simd']) & (df_updated_finn_estimate.idt == row1['idt']) & (df_updated_finn_estimate.wdt == row1['wdt']) & (df_updated_finn_estimate.act == row1['act']) & (df_updated_finn_estimate.mem_mode == row1['mem_mode'])]
        if not df_temp.empty:
            df_initial.at[index1, 'bram_new_finn_estimate'] = int(df_temp.iloc[0]['BRAM_new'])
    
    df_initial["Total_BRAM_18K_synth_denom"] = df_initial["Total_BRAM_18K synth"].apply(lambda x: 1 if x == 0 else x)
    df_initial["finn_rel_error_new"] = df_initial.apply(lambda x: (abs(x['bram_new_finn_estimate'] - x["Total_BRAM_18K synth"])/x["Total_BRAM_18K_synth_denom"])*100, axis=1)
    
    filepath_to_save = "../test_set_results/FCLayer/%s_updated.csv" % filename
    df_initial.to_csv(filepath_to_save, index = False, header=True)
    #import pdb; pdb.set_trace()

def plot_fclayer_models_test_set_results_bram():
    general_with_fu_model_file_name = "test_set_results_FCLayer_Total_BRAM_18K_general_plus_fu_updated"
    general_without_fu_model_file_name = "test_set_results_FCLayer_Total_BRAM_18K_general_min_fu_updated"
    specialized_without_fu_model_file_name = "test_set_results_FCLayer_Total_BRAM_18K_specialized_min_fu_updated"

    general_with_fu_model_folder_path = "../test_set_results/FCLayer/%s.csv" % general_with_fu_model_file_name
    general_without_fu_model_folder_path = "../test_set_results/FCLayer/%s.csv" % general_without_fu_model_file_name
    specialized_without_fu_model_folder_path = "../test_set_results/FCLayer/%s.csv" % specialized_without_fu_model_file_name

    df_general_with_fu_model = pd.read_csv(general_with_fu_model_folder_path)
    df_general_without_fu_model = pd.read_csv(general_without_fu_model_folder_path)
    df_specialized_without_fu_model = pd.read_csv(specialized_without_fu_model_folder_path)

    df_plot = pd.DataFrame()

    df_plot['HLS GM (FD)'] = df_general_with_fu_model['hls_rel_error']
    #df_plot['FINN GM (FD)'] = df_general_with_fu_model['finn_rel_error']
    #df_plot['FINN NEW GM (FD)'] = df_general_with_fu_model['finn_rel_error_new']
    df_plot['SVR GM (FD)'] = df_general_with_fu_model['svr_rel_error']

    df_plot['HLS GM (PD)'] = df_general_without_fu_model['hls_rel_error']
    #df_plot['FINN GM (PD)'] = df_general_without_fu_model['finn_rel_error']
    #df_plot['FINN NEW GM (PD)'] = df_general_without_fu_model['finn_rel_error_new']
    df_plot['SVR GM (PD)'] = df_general_without_fu_model['svr_rel_error']

    #df_plot['HLS SM (PD)'] = df_specialized_without_fu_model['hls_rel_error']
    #df_plot['FINN SM (PD)'] = df_specialized_without_fu_model['finn_rel_error']
    #df_plot['FINN NEW SM (PD)'] = df_specialized_without_fu_model['finn_rel_error_new']
    #df_plot['SVR SM (PD)'] = df_specialized_without_fu_model['svr_rel_error']

    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=False, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    colors = ['lightskyblue', 'lightyellow', 'lightskyblue', 'lightyellow']
    #colors = ['lightskyblue', 'lightyellow', 'lightskyblue', 'lightyellow', 'lightskyblue', 'lightyellow']
    #import pdb; pdb.set_trace()
    
    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation = 45)
    plt.title('FCLayer - BRAM estimation model - Test Set Results')
    plt.ylabel('Relative error [%] ')
    fig.savefig('../test_set_results/FCLayer/test_set_results_plot_bram_without_pd.png', bbox_inches='tight')

def plot_thresholding_models_test_set_results_lut():
    general_model_file_name = "test_set_results_Thresholding_LUT_general"
    general_augmentation_model_file_name = "test_set_results_Thresholding_LUT_general_augmentation"
    general_min_fu_model_file_name = "test_set_results_Thresholding_LUT_general_min_fu"
    general_min_fu_augmentation_model_file_name = "test_set_results_Thresholding_LUT_general_min_fu_augmentation"
    specialized_model_file_name = "test_set_results_Thresholding_LUT_specialized"
    specialized_min_fu_model_file_name = "test_set_results_Thresholding_LUT_specialized_min_fu"

    general_model_folder_path = "../test_set_results/Thresholding/%s.csv" % general_model_file_name
    general_augmentation_model_folder_path = "../test_set_results/Thresholding/%s.csv" % general_augmentation_model_file_name
    general_min_fu_model_folder_path = "../test_set_results/Thresholding/%s.csv" % general_min_fu_model_file_name
    general_min_fu_augmentation_model_folder_path = "../test_set_results/Thresholding/%s.csv" % general_min_fu_augmentation_model_file_name
    specialized_model_folder_path = "../test_set_results/Thresholding/%s.csv" % specialized_model_file_name
    specialized_min_fu_model_folder_path = "../test_set_results/Thresholding/%s.csv" % specialized_min_fu_model_file_name

    df_general_model = pd.read_csv(general_model_folder_path)
    df_general_augmentation_model = pd.read_csv(general_augmentation_model_folder_path)
    df_general_min_fu_model = pd.read_csv(general_min_fu_model_folder_path)
    df_general_min_fu_augmentation_model = pd.read_csv(general_min_fu_augmentation_model_folder_path)
    df_specialized_model = pd.read_csv(specialized_model_folder_path)
    df_specialized_min_fu_model = pd.read_csv(specialized_min_fu_model_folder_path)

    df_plot = pd.DataFrame()
    df_plot['HLS GM (FD)'] = df_general_model['hls_rel_error']
    df_plot['FINN GM (FD)'] = df_general_model['finn_rel_error']
    df_plot['SVR GM (FD)'] = df_general_model['svr_rel_error']
    
    #df_plot['HLS GM (FD + AUG)'] = df_general_augmentation_model['hls_rel_error']
    #df_plot['FINN GM (FD + AUG)'] = df_general_augmentation_model['finn_rel_error']
    #df_plot['SVR GM (FD + AUG)'] = df_general_augmentation_model['svr_rel_error']
    
    df_plot['HLS GM (PD)'] = df_general_min_fu_model['hls_rel_error']
    df_plot['FINN GM (PD)'] = df_general_min_fu_model['finn_rel_error']
    df_plot['SVR GM (PD)'] = df_general_min_fu_model['svr_rel_error']

    #df_plot['HLS GM (PD)'] = df_general_min_fu_model['hls_rel_error']
    #df_plot['FINN GM (PD)'] = df_general_min_fu_model['finn_rel_error']
    #df_plot['SVR GM (PD + AUG)'] = df_general_min_fu_augmentation_model['svr_rel_error']

    #df_plot['HLS SM (FD)'] = df_specialized_model['hls_rel_error']
    #df_plot['FINN SM (FD)'] = df_specialized_model['finn_rel_error']
    #df_plot['SVR SM (FD)'] = df_specialized_model['svr_rel_error']

    df_plot['HLS SM (PD)'] = df_specialized_min_fu_model['hls_rel_error']
    df_plot['FINN SM (PD)'] = df_specialized_min_fu_model['finn_rel_error']
    df_plot['SVR SM (PD)'] = df_specialized_min_fu_model['svr_rel_error']

    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=False, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    #colors = ['lightskyblue', 'lightgreen', 'lightyellow', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow']
    colors = ['lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow']
    
    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(rotation = 45)
    plt.title('Thresholding Layer - LUT estimation model - Test Set Results')
    plt.ylabel('Relative error [%] ')
    fig.savefig('../test_set_results/Thresholding/test_set_results_LUT_plot_without_outliers_plus_fu.png', bbox_inches='tight')

def plot_thresholding_models_test_set_results_bram():

    general_model_file_name = "test_set_results_Thresholding_Total_BRAM_18K_general"
    general_min_fu_model_file_name = "test_set_results_Thresholding_Total_BRAM_18K_general_min_fu"

    general_model_folder_path = "../test_set_results/Thresholding/%s.csv" % general_model_file_name
    general_min_fu_model_folder_path = "../test_set_results/Thresholding/%s.csv" % general_min_fu_model_file_name

    df_general_model = pd.read_csv(general_model_folder_path)
    df_general_min_fu_model = pd.read_csv(general_min_fu_model_folder_path)

    df_plot = pd.DataFrame()
    df_plot['HLS GM (FD)'] = df_general_model['hls_rel_error']
    df_plot['FINN GM (FD)'] = df_general_model['finn_rel_error']
    df_plot['SVR GM (FD)'] = df_general_model['svr_rel_error']
    
    df_plot['HLS GM (PD)'] = df_general_min_fu_model['hls_rel_error']
    df_plot['FINN GM (PD)'] = df_general_min_fu_model['finn_rel_error']
    df_plot['SVR GM (PD)'] = df_general_min_fu_model['svr_rel_error']
    
    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=True, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    colors = ['lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow']
    
    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(rotation = 45)
    plt.title('Thresholding Layer - BRAM estimation model - Test Set Results')
    plt.ylabel('Relative error [%] ')
    fig.savefig('../test_set_results/Thresholding/test_set_results_BRAM_plot_with_outliers.png', bbox_inches='tight')

def plot_SWU_models_test_set_results_lut():
    general_model_file_name = "test_set_results_Sliding_Window_Unit_LUT_general"
    specialized_dw_0_model_file_name = "test_set_results_Sliding_Window_Unit_LUT_specialized_dw_0"
    specialized_dw_1_model_file_name = "test_set_results_Sliding_Window_Unit_LUT_specialized_dw_1"

    general_model_folder_path = "../test_set_results/Sliding_Window_Unit/%s.csv" % general_model_file_name
    specialized_dw_0_model_folder_path = "../test_set_results/Sliding_Window_Unit/%s.csv" % specialized_dw_0_model_file_name
    specialized_dw_1_model_folder_path = "../test_set_results/Sliding_Window_Unit/%s.csv" % specialized_dw_1_model_file_name

    df_general_model = pd.read_csv(general_model_folder_path)
    df_specialized_dw_0_model = pd.read_csv(specialized_dw_0_model_folder_path)
    df_specialized_dw_1_model = pd.read_csv(specialized_dw_1_model_folder_path)

    df_plot = pd.DataFrame()
    #df_plot['HLS GM'] = df_general_model['hls_rel_error']
    #df_plot['FINN GM'] = df_general_model['finn_rel_error']
    df_plot['SVR GM'] = df_general_model['svr_rel_error']

    #df_plot['HLS SM (dw=0)'] = df_specialized_dw_0_model['hls_rel_error']
    #df_plot['FINN SM (dw=0)'] = df_specialized_dw_0_model['finn_rel_error']
    df_plot['SVR SM (dw=0)'] = df_specialized_dw_0_model['svr_rel_error']

    #df_plot['HLS SM (dw=1)'] = df_specialized_dw_1_model['hls_rel_error']
    #df_plot['FINN SM (dw=1)'] = df_specialized_dw_1_model['finn_rel_error']
    df_plot['SVR SM (dw=1)'] = df_specialized_dw_1_model['svr_rel_error']

    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=False, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    #colors = ['lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow']
    colors = ['lightyellow', 'lightyellow', 'lightyellow']

    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation = 45)
    plt.title('Sliding Window Unit - LUT estimation model - Test Set Results')
    plt.ylabel('Relative error [%] ')
    fig.savefig('../test_set_results/Sliding_Window_Unit/test_set_results_LUT_plot_only_svr.png', bbox_inches='tight')

def plot_SWU_models_test_set_results_bram():
    general_model_file_name = "test_set_results_Sliding_Window_Unit_Total_BRAM_18K_general"
    specialized_dw_0_model_file_name = "test_set_results_Sliding_Window_Unit_Total_BRAM_18K_specialized_dw_0"
    specialized_dw_1_model_file_name = "test_set_results_Sliding_Window_Unit_Total_BRAM_18K_specialized_dw_1"

    general_model_folder_path = "../test_set_results/Sliding_Window_Unit/%s.csv" % general_model_file_name
    specialized_dw_0_model_folder_path = "../test_set_results/Sliding_Window_Unit/%s.csv" % specialized_dw_0_model_file_name
    specialized_dw_1_model_folder_path = "../test_set_results/Sliding_Window_Unit/%s.csv" % specialized_dw_1_model_file_name

    df_general_model = pd.read_csv(general_model_folder_path)
    df_specialized_dw_0_model = pd.read_csv(specialized_dw_0_model_folder_path)
    df_specialized_dw_1_model = pd.read_csv(specialized_dw_1_model_folder_path)

    df_plot = pd.DataFrame()
    df_plot['HLS GM'] = df_general_model['hls_rel_error']
    df_plot['FINN GM'] = df_general_model['finn_rel_error']
    df_plot['SVR GM'] = df_general_model['svr_rel_error']

    df_plot['HLS SM (dw=0)'] = df_specialized_dw_0_model['hls_rel_error']
    df_plot['FINN SM (dw=0)'] = df_specialized_dw_0_model['finn_rel_error']
    df_plot['SVR SM (dw=0)'] = df_specialized_dw_0_model['svr_rel_error']

    df_plot['HLS SM (dw=1)'] = df_specialized_dw_1_model['hls_rel_error']
    df_plot['FINN SM (dw=1)'] = df_specialized_dw_1_model['finn_rel_error']
    df_plot['SVR SM (dw=1)'] = df_specialized_dw_1_model['svr_rel_error']

    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=True, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    colors = ['lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightskyblue', 'lightgreen', 'lightyellow']
    
    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation = 45)
    plt.title('Sliding Window Unit - BRAM estimation model - Test Set Results')
    plt.ylabel('Relative error [%] ')
    fig.savefig('../test_set_results/Sliding_Window_Unit/test_set_results_BRAM_plot_with_outliers.png', bbox_inches='tight')

def plot_fclayer_target_processing_results():
    prep_none_file_name = "test_set_results_FCLayer_LUT_general_preprocessing_none"
    prep_log_file_name = "test_set_results_FCLayer_LUT_general_preprocessing_log"
    prep_diff_file_name = "test_set_results_FCLayer_LUT_general_preprocessing_diff"

    prep_none_folder_path = "../test_set_results/FCLayer/%s.csv" % prep_none_file_name
    prep_log_folder_path = "../test_set_results/FCLayer/%s.csv" % prep_log_file_name
    prep_diff_folder_path = "../test_set_results/FCLayer/%s.csv" % prep_diff_file_name

    df_prep_none = pd.read_csv(prep_none_folder_path)
    df_prep_log = pd.read_csv(prep_log_folder_path)
    df_prep_diff = pd.read_csv(prep_diff_folder_path)

    df_plot = pd.DataFrame()
    #df_plot['HLS GM'] = df_prep_none['hls_rel_error']
    #df_plot['FINN GM'] = df_prep_none['finn_rel_error']
    df_plot['None'] = df_prep_none['svr_rel_error']

    #df_plot['HLS SM (dw=0)'] = df_prep_log['hls_rel_error']
    #df_plot['FINN SM (dw=0)'] = df_prep_log['finn_rel_error']
    df_plot['LOG'] = df_prep_log['svr_rel_error']

    #df_plot['HLS SM (dw=1)'] = df_prep_diff['hls_rel_error']
    #df_plot['FINN SM (dw=1)'] = df_prep_diff['finn_rel_error']
    df_plot['DIFF (SYNTH(ground truth) - FINN estimate)'] = df_prep_diff['svr_rel_error']

    fig = plt.figure(figsize=(20, 11))
    boxplot = df_plot.boxplot(showmeans=True, showfliers=False, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)
    
    colors = ['lightyellow', 'lightyellow', 'lightyellow']
    
    for patch, color in zip(boxplot['means'], colors):
        patch.set_markeredgecolor('red')
        patch.set_markerfacecolor('red')
        
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    #plt.xticks(rotation = 45)
    plt.title('Comparison between different methods of preprocessing')
    plt.ylabel('Relative error [%] ')
    fig.savefig('../test_set_results/FCLayer/test_set_results_preprocessing.png', bbox_inches='tight')

#add_newest_finn_estimation_to_the_csv_file("test_set_results_FCLayer_Total_BRAM_18K_general_plus_fu")
#add_newest_finn_estimation_to_the_csv_file("test_set_results_FCLayer_Total_BRAM_18K_general_min_fu")
#add_newest_finn_estimation_to_the_csv_file("test_set_results_FCLayer_Total_BRAM_18K_specialized_min_fu")

#plot_fclayer_models_test_set_results_lut()
#plot_fclayer_models_test_set_results_bram()
#plot_thresholding_models_test_set_results_lut()
#plot_thresholding_models_test_set_results_bram()
#plot_SWU_models_test_set_results_lut()
#plot_SWU_models_test_set_results_bram()

plot_fclayer_target_processing_results()