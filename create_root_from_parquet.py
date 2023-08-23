import subprocess
import os
import json
import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import pandas as pd
# function for change parquet to root ntuples for flashgg final fit
def create_root_ntuples(cat, inputpath, input_name, outputpath, output_tree_name, output_root_name, ifmc, cut_var, cut_var_value):
    # 检查文件夹是否存在
    if not os.path.exists(outputpath):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(outputpath)
        print(f"Folder '{outputpath}' created successfully.")
    # load events
    events = ak.from_parquet(inputpath + input_name)
    # choose category
    # 7 catgory
    # 0: no category
    # 1: SL_boosted_cat
    # 2: SL_fullyresovled_cat
    # 3: SL_merged_boosted_cat
    # 4: FH_boosted
    # 5: FH_2Wfatjet_cat
    # 6: FH_1Wfatjet_cat
    # 7: FH_fully_resovled_cat
    if ('no_cat' in cat):
        category_cut = ((events.category == 0) & (events.Diphoton_minID > -0.7))
    elif ('SL_boosted_cat' in cat):
        category_cut = ((events.category == 1)& (events.Diphoton_minID > -0.7))
    elif ('SL_fullyresovled_cat' in cat):
        category_cut = ((events.category == 2)& (events.Diphoton_minID > -0.7))
    elif ('SL_merged_boosted_cat' in cat):
        category_cut = ((events.category == 3)& (events.Diphoton_minID > -0.7)) 
    elif ('FH_boosted' in cat):
        category_cut = ((events.category == 4)& (events.Diphoton_minID > -0.7))
    elif ('FH_2Wfatjet_cat' in cat):
        category_cut = ((events.category == 5)& (events.Diphoton_minID > -0.7))
    elif ('FH_1Wfatjet_cat' in cat):
        category_cut = ((events.category == 6)& (events.Diphoton_minID > -0.7))
    elif ('FH_fully_resovled_cat' in cat):
        category_cut = ((events.category == 7)& (events.Diphoton_minID > -0.7))
    else:
        print("Wrong category name")
    events_cat = events[category_cut]
    events_cat = events_cat[np.logical_and(getattr(events_cat,cut_var)>cut_var_value[0], getattr(events_cat,cut_var)<=cut_var_value[1]) ]
    # change vars name for flashggfinalfit
    events_cat["CMS_hgg_mass"] = events_cat["Diphoton_mass"]
    if (ifmc == 1):
        events_cat['weight'] = events_cat['weight_central_no_lumi']
    else:
        events_cat['weight'] = events_cat['weight_central']
    events_cat["dZ"] = np.ones(len(events_cat['Diphoton_mass']))
    # to parquet to transform to root
    # save dataframe to parquet
    # events_cat.to_parquet(events_cat,inputpath+"merged_nominal_changed.parquet")
    ak.to_parquet(events_cat,inputpath+"merged_nominal_changed.parquet")
    # parquet to root
    from parquet_to_root import parquet_to_root
    parquet_to_root(inputpath+"merged_nominal_changed.parquet", outputpath + output_root_name, treename=output_tree_name, verbose=False)


# for data with all categories
cat_list = ['FHSamples_SL_boosted_cat','FHSamples_SL_fullyresovled_cat','FHSamples_SL_merged_boosted_cat','FHSamples_FH_boosted','FHSamples_FH_2Wfatjet_cat','FHSamples_FH_1Wfatjet_cat','FHSamples_FH_fully_resovled_cat']
# cat_list = ['FHSamples_SL_boosted_cat','FHSamples_SL_fullyresovled_cat','FHSamples_SL_merged_boosted_cat','FHSamples_FH_2Wfatjet_cat','FHSamples_FH_1Wfatjet_cat','FHSamples_FH_fully_resovled_cat']
# mass_list = ["M1600","M1800","M1900","M2000","M2200","M2400","M2600","M2800","M3000","M1100","M1200","M1300","M1400","M1500","M1700"]
mass_list = ["M1100"]
# mass_path = ["UL17_R_gghh_M-1600_2017","UL17_R_gghh_M-1800_2017","UL17_R_gghh_M-1900_2017","UL17_R_gghh_M-2000_2017","UL17_R_gghh_M-2200_2017","UL17_R_gghh_M-2400_2017","UL17_R_gghh_M-2600_2017","UL17_R_gghh_M-2800_2017","UL17_R_gghh_M-3000_2017","UL17_R_gghh_M-1100_2017","UL17_R_gghh_M-1200_2017","UL17_R_gghh_M-1300_2017","UL17_R_gghh_M-1400_2017","UL17_R_gghh_M-1500_2017","UL17_R_gghh_M-1700_2017"]
mass_path = ["UL17_R_gghh_M-1100_2017"]
for i in range(len(mass_list)):
    mass = mass_list[i]
    for cat in cat_list:
        output_path_data = "/eos/user/z/zhenxuan/hhwwgg_workspace/hhwwgg_root/hhwwgg_root_FH/" + mass + "/"
        output_tree_name_data = "Data_13TeV_RECO_untagged_" + cat
        output_root_name_data = "Data_2017_" + cat + "_" +mass+".root"
        input_path_data = "/eos/user/s/shsong/HHWWgg/parquet/data/"

        create_root_ntuples(cat=cat, inputpath=input_path_data, input_name = "merged_nominal.parquet", outputpath = output_path_data, output_tree_name=output_tree_name_data , output_root_name=output_root_name_data, cut_var='fatjet_H_Hqqqq_vsQCDTop', cut_var_value=[-10000,10000], ifmc=0)

        input_path_signal = "/eos/user/s/shsong/HHWWgg/parquet/FH_1200to3000/" + mass_path[i] + "/"
        output_path_signal = "/eos/user/z/zhenxuan/hhwwgg_workspace/hhwwgg_root/hhwwgg_root_FH/" + mass + "/"
        output_tree_name_signal = "gghh_125_13TeV_RECO_untagged_" + cat 
        output_root_name_signal = "Signal_"+mass + "_2017_" + cat +".root"

        create_root_ntuples(cat=cat, inputpath=input_path_signal,input_name = "merged_nominal.parquet", outputpath = output_path_signal, output_tree_name=output_tree_name_signal, output_root_name=output_root_name_signal, cut_var='fatjet_H_Hqqqq_vsQCDTop', cut_var_value=[-10000,10000], ifmc=1)

# load categories from JSON file
# with open("categories_all_mass_point.json", "r") as f:
    # categories_dict = json.load(f)

# get list of signal points from categories dictionary
# list_of_sig = list(categories_dict.keys())

# loop over signal points and categories to create ROOT ntuples
# for sig_index in range(len(list_of_sig)):
#     # get boundaries for categories of this signal point
#     boundary_list = categories_dict[list_of_sig[sig_index]]
#     for i in range(len(boundary_list)):
#         create_root_ntuples(cat='1jet', inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/FHSL_channel/",input_name = "UL17_R_gghh_FHSL_" + list_of_sig[sig_index]+ "_2017_cat1.parquet", outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_of_sig[sig_index] + "/", output_tree_name="gghh_125_13TeV_RECO_untagged_1jets"+"_cat" + str(i), output_root_name="Signal_"+list_of_sig[sig_index] + "_FHSL_2017_1jets"+"_cat" + str(i)+".root", cut_var='fatjet_H_Hqqqq_qqlv_vsQCDTop', cut_var_value=boundary_list[i], ifmc=1)
#         create_root_ntuples(cat='1jet', inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/Data/hhwwgg_data_FHSL_cat1/", input_name = "merged_nominal.parquet", outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_of_sig[sig_index] + "/", output_tree_name="Data_13TeV_RECO_untagged_1jets"+"_cat" + str(i), output_root_name="Data_FHSL_2017_cat_1jets"+"_cat" + str(i)+ "_" +list_of_sig[sig_index]+".root", cut_var='fatjet_H_Hqqqq_qqlv_vsQCDTop', cut_var_value=boundary_list[i], ifmc=0)

# for i in range(100,1001,100):
#     # try to loop 100~1000 diphoton pt
#     __output_tree_name = "Data_13TeV_RECO_untagged_1jets_diphopt_l"+str(i)
#     __output_root_name = "Data_FH_2017_cat_1jets_diphopt_l"+str(i)+".root"
#     print("output_tree_name:",__output_tree_name)
#     print("output_root_name:",__output_root_name)
    # create_root_ntuples(inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/FH_channel/hhwwgg_data_FH_custom_v2/",outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom_v2/", output_tree_name=__output_tree_name, cat='1jet', output_root_name=__output_root_name, cut_var='Diphoton_pt', cut_var_value=i, ifmc=0)
#     # signal
#     __output_tree_name = "Data_13TeV_RECO_untagged_1jets_diphopt_l"+str(i)
#     __output_root_name = "Data_FH_2017_cat_1jets_diphopt_l"+str(i)+".root"
#     print("output_tree_name:",__output_tree_name)
#     print("output_root_name:",__output_root_name)
    # create_root_ntuples(inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/FH_channel/hhwwgg_signal_FH_custom/UL17_R_gghh_M-3000_2017/",outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom_v2/", output_tree_name="gghh_125_13TeV_RECO_untagged_1jets", cat='1jets', output_root_name="Signal_M3000_FH_2017_1jets.root", ifmc=1,cut_var='Diphoton_mass', cut_var_value=300)


# signal