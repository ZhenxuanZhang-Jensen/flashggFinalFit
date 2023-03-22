import subprocess
import os
import json
import ROOT
import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
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
    if (cat == '1jet'):
        category_cut = ((events.category == 1))
    elif(cat == '4jets'):
        category_cut = ((events.category == 3))
    events_cat = events[category_cut]
    events_cat = events_cat[np.logical_and(getattr(events_cat,cut_var)>cut_var_value[0], getattr(events_cat,cut_var)<cut_var_value[1]) ]
    # change vars name for flashggfinalfit
    events_cat["CMS_hgg_mass"] = events_cat["Diphoton_mass"]
    if (ifmc == 1):
        events_cat['weight'] = events_cat['weight_central_no_lumi']
    else:
        events_cat['weight'] = events_cat['weight_central']
    events_cat["dZ"] = np.ones(len(events_cat['Diphoton_mass']))
    # to parquet to transform to root
    ak.to_parquet(events_cat,inputpath+"merged_nominal_changed.parquet")
    # parquet to root
    from parquet_to_root import parquet_to_root
    parquet_to_root(inputpath+"merged_nominal_changed.parquet", outputpath + output_root_name, treename=output_tree_name, verbose=False)


# load categories from JSON file
with open("categories.json", "r") as f:
    categories_dict = json.load(f)

# get list of signal points from categories dictionary
list_of_sig = list(categories_dict.keys())

# loop over signal points and categories to create ROOT ntuples
for sig_index in range(len(list_of_sig)):
    # get boundaries for categories of this signal point
    boundary_list = categories_dict[list_of_sig[sig_index]]
    for i in range(len(boundary_list)):
        create_root_ntuples(cat='1jet', inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/FHSL_channel/",input_name = "UL17_R_gghh_FHSL_" + list_of_sig[sig_index]+ "_2017_cat1.parquet", outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_of_sig[sig_index] + "/", output_tree_name="gghh_125_13TeV_RECO_untagged_1jets"+"_cat" + str(i), output_root_name="Signal_"+list_of_sig[sig_index] + "_FHSL_2017_1jets"+"_cat" + str(i)+".root", cut_var='fatjet_H_Hqqqq_qqlv_vsQCDTop', cut_var_value=boundary_list[i], ifmc=1)
        create_root_ntuples(cat='1jet', inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/Data/hhwwgg_data_FHSL_cat1/", input_name = "merged_nominal.parquet", outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_of_sig[sig_index] + "/", output_tree_name="Data_13TeV_RECO_untagged_1jets"+"_cat" + str(i), output_root_name="Data_FHSL_2017_cat_1jets"+"_cat" + str(i)+ "_" +list_of_sig[sig_index]+".root", cut_var='fatjet_H_Hqqqq_qqlv_vsQCDTop', cut_var_value=boundary_list[i], ifmc=0)

# for i in range(100,1001,100):
#     # try to loop 100~1000 diphoton pt
#     __output_tree_name = "Data_13TeV_RECO_untagged_1jets_diphopt_l"+str(i)
#     __output_root_name = "Data_FH_2017_cat_1jets_diphopt_l"+str(i)+".root"
#     print("output_tree_name:",__output_tree_name)
#     print("output_root_name:",__output_root_name)
#     # create_root_ntuples(inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/FH_channel/hhwwgg_data_FH_custom_v2/",outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom_v2/", output_tree_name=__output_tree_name, cat='1jet', output_root_name=__output_root_name, cut_var='Diphoton_pt', cut_var_value=i, ifmc=0)
#     # signal
#     __output_tree_name = "Data_13TeV_RECO_untagged_1jets_diphopt_l"+str(i)
#     __output_root_name = "Data_FH_2017_cat_1jets_diphopt_l"+str(i)+".root"
#     print("output_tree_name:",__output_tree_name)
#     print("output_root_name:",__output_root_name)
#     create_root_ntuples(inputpath="/eos/user/z/zhenxuan/hhwwgg_parquet/FH_channel/hhwwgg_signal_FH_custom/UL17_R_gghh_M-3000_2017/",outputpath = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom_v2/", output_tree_name="gghh_125_13TeV_RECO_untagged_1jets", cat='1jets', output_root_name="Signal_M3000_FH_2017_1jets.root", ifmc=1,cut_var='Diphoton_mass', cut_var_value=300)


# signal