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
        print("no_cat")
        category_cut = ((events.category == 0) & (events.Diphoton_minID > -0.7))
    elif ('SL_boosted_cat' in cat):
        print("SL_boosted_cat")
        category_cut = ((events.category == 1)& (events.Diphoton_minID > -0.7))
    elif ('SL_fullyresovled_cat' in cat):
        print("SL_fullyresovled_cat")
        category_cut = ((events.category == 2)& (events.Diphoton_minID > -0.7))
    elif ('SL_merged_boosted_cat' in cat):
        print("SL_merged_boosted_cat")
        # electron channel
        is_electron = ak.where(events["electron_noniso_pt"] != -999, True, False)
        # muon channel
        is_muon = ak.where(events["muon_noniso_pt"] != -999, True, False)
        # add noniso lepton pt cut
        noniso_lepton_pt = ak.where(events["electron_noniso_pt"] != -999, events["electron_noniso_pt"], events["muon_noniso_pt"])
        category_cut = ((events.category == 3)& (events.Diphoton_minID > -0.7) & (noniso_lepton_pt > 70) & (events['Diphoton_pt']>200)) 
    elif ('FH_boosted' in cat):
        print("FH_boosted")
        category_cut = ((events.category == 4)& (events.Diphoton_minID > -0.7))
    elif ('FH_2Wfatjet_cat' in cat):
        print("FH_2Wfatjet_cat")
        category_cut = ((events.category == 5)& (events.Diphoton_minID > -0.7))
    elif ('FH_1Wfatjet_cat' in cat):
        print("FH_1Wfatjet_cat")
        category_cut = ((events.category == 6)& (events.Diphoton_minID > -0.7))
    elif ('FH_fully_resovled_cat' in cat):
        print("FH_fully_resovled_cat")
        category_cut = ((events.category == 7)& (events.Diphoton_minID > -0.7))
    else:
        print("Wrong category name")
    events_cat = events[category_cut]
    events_cat = events_cat[np.logical_and(getattr(events_cat,cut_var)>cut_var_value[0], getattr(events_cat,cut_var)<=cut_var_value[1]) ]
    # print number of events
    print("Number of events in category ", str(len(events_cat)) )
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
    ak.to_parquet(events_cat,inputpath+"FH_merged_nominal_changed.parquet")
    # parquet to root
    from parquet_to_root import parquet_to_root
    parquet_to_root(inputpath+"FH_merged_nominal_changed.parquet", outputpath + output_root_name, treename=output_tree_name, verbose=False)


# for data with all categories
# cat_list = ['FHSamples_SL_boosted_cat','FHSamples_SL_fullyresovled_cat','FHSamples_SL_merged_boosted_cat','FHSamples_FH_boosted','FHSamples_FH_2Wfatjet_cat','FHSamples_FH_1Wfatjet_cat','FHSamples_FH_fully_resovled_cat']
# cat_list = ['FHSamples_SL_boosted_cat','FHSamples_SL_fullyresovled_cat','FHSamples_SL_merged_boosted_cat','FHSamples_FH_boosted','FHSamples_FH_2Wfatjet_cat','FHSamples_FH_1Wfatjet_cat','FHSamples_FH_fully_resovled_cat']
# cat_list = ['FHSLCombine_FH_boosted_low_purity', "FHSLCombine_FH_boosted_high_purity"]
# cat_list = ['FHSLCombine_SL_boosted_cat','FHSLCombine_SL_fullyresovled_cat','FHSLCombine_SL_merged_boosted_cat','FHSLCombine_FH_2Wfatjet_cat','FHSLCombine_FH_1Wfatjet_cat','FHSLCombine_FH_fully_resovled_cat','FHSLCombine_FH_boosted_low_purity', "FHSLCombine_FH_boosted_high_purity"]
cat_list = ['FHSLCombine_SL_merged_boosted_cat']
# mass_list = ['M1100_FH', 'M1200_FH', 'M1300_FH', 'M1400_FH', 'M1500_FH', 'M1600_FH', 'M1700_FH', 'M1800_FH', 'M1900_FH', 'M2000_FH', 'M2100_FH', 'M2200_FH', 'M2300_FH', 'M2400_FH', 'M2500_FH', 'M2600_FH', 'M2700_FH', 'M2800_FH', 'M2900_FH', 'M3000_FH']
# mass_list = ["M1200_FH","M1300_FH","M1400_FH","M1500_FH","M1700_FH","M1600_FH","M1800_FH","M1900_FH","M2000_FH","M2200_FH","M2400_FH","M2600_FH","M2800_FH"]
# mass_list = ['M1100_FH']
# mass_list = ["M600_FH","M650_FH","M700_FH","M750_FH","M850_FH","M900_FH","M1000_FH","M800_FH"]
# mass_list =["M250_FH","M260_FH","M280_FH","M300_FH","M320_FH","M270_FH","M350_FH","M400_FH","M450_FH","M550_FH"]
mass_list = ["Mtest_FH"]
# mass_path= ["/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1100_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1200_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1300_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1400_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1500_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1700_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1600_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1800_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-1900_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-2000_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-2200_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-2400_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-2600_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-2800_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_1100to3000/UL17_R_gghh_M-3000_2017"]
# mass_path = ["/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-600_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-650_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-700_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-750_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-850_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-900_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-1000_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-800_2017/merged_nominal.parquet"]
# mass_path = ["/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-250_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-260_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-280_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-300_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-320_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-270_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-350_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-400_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-450_2017","/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-550_2017"]
mass_path= ["/eos/user/s/shsong/HHWWgg_omitdiphotonpt_tightID/parquet/FH_1200to3000/UL17_R_gghh_M-3000_2017"]
signal_process = "gghhFH"
for i in range(len(mass_list)):
    mass = mass_list[i]
    for cat in cat_list:
        if "low" in cat:
            cut_list = [0.4, 0.75]
        elif "high" in cat:
            cut_list = [0.75, 1.0]
        else:
            cut_list = [-100000, 1000000]
        print("cut_list", cut_list)
        output_path_data = "/eos/user/z/zhenxuan/hhwwgg_workspace/hhwwgg_root/hhwwgg_root_FHSL/" + mass + "/" 
        output_tree_name_data = "Data_13TeV_RECO_untagged_" + cat
        output_root_name_data = "Data_2017_" + cat + "_" +mass+".root"
        # input_path_data = "/eos/user/s/shsong/HHWWgg_omitdiphotonpt_tightID/parquet/data/" # old one with diphoton pt > 300
        input_path_data = "/eos/user/s/shsong/HHWWgg_omitdiphotonpt_tightID/parquet/data/" # new one without diphoton pt cut

        create_root_ntuples(cat=cat, inputpath=input_path_data, input_name = "merged_nominal.parquet", outputpath = output_path_data, output_tree_name=output_tree_name_data , output_root_name=output_root_name_data, cut_var='fatjet_H_Hqqqq_vsQCDTop', cut_var_value=cut_list, ifmc=0)

        input_path_signal = mass_path[i] + "/"
        output_path_signal = "/eos/user/z/zhenxuan/hhwwgg_workspace/hhwwgg_root/hhwwgg_root_FHSL/" + mass + "/"
        output_tree_name_signal = signal_process + "_125_13TeV_RECO_untagged_" + cat 
        output_root_name_signal = "Signal_"+mass + "_2017_" + cat +".root"

        create_root_ntuples(cat=cat, inputpath=input_path_signal,input_name = "merged_nominal.parquet", outputpath = output_path_signal, output_tree_name=output_tree_name_signal, output_root_name=output_root_name_signal, cut_var='fatjet_H_Hqqqq_vsQCDTop', cut_var_value=cut_list, ifmc=1)
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