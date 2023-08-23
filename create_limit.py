# -*- coding: utf-8 -*-
'''
Hi my friend, believe it or not, I just want to create a king script to create limit with flashggfinal fit framework from ntuples(root) with only one script
let see if the magic can work
'''
import codecs
import uproot 
import awkward as ak
import json
from collections import defaultdict
# import ROOT
import os 
import logging
import sys
import subprocess
import time
logging.basicConfig(filename='/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/logging_output.log', level=logging.INFO, filemode="w")
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
# ----------------------------  trees2WS part ---------------------------- #
def run_Tree2WS_sig(inputpath_name, inputfile_name, log_file_name, ws_path, output_sig_root_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Trees2WS")
    # no need to change the config file since all are auto set
    command = " python trees2ws.py --inputConfig config_simple.py --inputTreeFile " + inputpath_name+inputfile_name + " --inputMass 125 --productionMode gghh --year 2017  > " + log_file_name+ " 2>&1"
    logging.info("the Tree2WS command:")
    logging.info(command)
    # run trees2ws at shell
    run_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("end: {}".format(time.time()))
    # create the dir for this workspace
    command = "mkdir " +inputpath_name+ "ws_gghh_" + ws_path
    mkdir_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("mkdir for ws \n command :{0}".format(command))
    # cp the root file to the ws dir
    command = "cp " + inputpath_name +"ws_gghh/" + inputfile_name.split('.root')[0] + '_gghh' + ".root" + " " + inputpath_name +"ws_gghh_"+ws_path +"/" + output_sig_root_name
    cp_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("cp to ws dir \n command: {0}".format(command))
def run_Tree2WS_data(inputpath_name, ws_data_path, output_data_root_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Trees2WS")
    # run data tree2ws
    command = "python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile " + inputpath_name + output_data_root_name
    run_data_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    # mkdir 
    command = "mkdir " +inputpath_name+  ws_data_path
    mkdir_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    # run cp data 
    command = "cp " + inputpath_name +"ws/" + output_data_root_name + " " + inputpath_name + ws_data_path +"/" + "allData.root"
    cp_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("cp to ws dir \n command: {0}".format(command))

# ------------------------------ Signal fit part ----------------------------- #
def run_ftest(ws_path, log_name, inputpath_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Signal")
    # modify the config file
    #copy config_toy.py
    command = "cp config_toy.py " + "config_" + ws_path + ".py"
    run_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("copy config_toy.py \n command :{0}".format(command))
    time.sleep(2)
    # sed config_*.py
    command1 = 'sed -i "s#ws_path#' + ws_path + '#g" ' +  "config_" + ws_path + ".py" 
    sed1_p = subprocess.call(command1, shell=True)
    logging.info("sed1 config_*.py \n command :{0}".format(command1))

    command2 = 'sed -i "s#input_path#' + inputpath_name + '#g" ' + "config_" + ws_path + ".py"

    sed2_p = subprocess.call(command2, shell=True)
    logging.info("sed2 config_*.py \n command :{0}".format(command2))
    # run ftest
    command = "python RunSignalScripts.py --inputConfig " +  "config_" + ws_path + ".py"+ " --mode 'fTest'" + " > " + log_name + " 2>&1"
    run_ftest_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run ftest \n command :{0}".format(command))
    logging.info("end: {}".format(time.time()))

def run_signalfit(ws_path, log_name, inputpath_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Signal")
    # run signalfit
    command = "python RunSignalScripts.py --inputConfig " +  "config_" + ws_path + ".py"+ " --mode 'signalFit' --modeOpts '--skipSystematics' " + " > " + log_name + " 2>&1"
    logging.info("run signalfit \n command :{0}".format(command))
    run_signalfit_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    # cp the signalfit output root file to ws
    command = "cp outdir_dcb_2017_" + ws_path + "/signalFit/output/*.root" + " " + inputpath_name +  "ws_gghh_"+ws_path
    cp_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run cp the signalfit output root file to ws \n command :{0}".format(command))

    logging.info("end: {}".format(time.time()))

def run_signal_plot(outputExt, cats, exts,log_packaged_name,ws_path,inputpath_name,log_plotter_name, cp_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Signal")
    # run packaged
    command = "python RunPackager.py --cats " + cats + " --exts " + exts + "  --batch local  --massPoints 125 --year 2017 --outputExt " + outputExt + " > " + log_packaged_name + " 2>&1"
    run_packaged_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run packaged \n command :{0}".format(command))
    # run plotter
    command = "python RunPlotter.py --cats " + cats + " --procs all --years 2017 --ext " + outputExt + " > " + log_plotter_name + " 2>&1"
    run_plotter_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run plotter \n command :{0}".format(command))
    # cp the signalfit output root file to ws
    command = "cp outdir_packaged_" + ws_path + "/*.root" + " " + inputpath_name +  "ws_gghh_"+ ws_path + "/" + cp_name
    cp_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run cp the signalfit output root file to ws \n command :{0}".format(command))
    logging.info("end: {}".format(time.time()))
# ---------------------------- Background fit part --------------------------- #
def run_backgroundfit(ws_data_path, log_name, inputpath_name, ext_name, cp_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Background")
    # modify the config file
    #copy config_toy.py
    command = "cp config_toy.py " + "config_" + ws_data_path + ".py"
    run_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("copy config_toy.py \n command :{0}".format(command))
    time.sleep(2)
    # sed config_*.py
    command1 = 'sed -i "s#ws_path#' + ws_data_path + '#g" ' +  "config_" + ws_data_path + ".py" 
    sed1_p = subprocess.call(command1, shell=True)
    logging.info("sed1 config_*.py \n command :{0}".format(command1))

    command2 = 'sed -i "s#input_path#' + inputpath_name + '#g" ' + "config_" + ws_data_path + ".py"

    sed2_p = subprocess.call(command2, shell=True)
    logging.info("sed2 config_*.py \n command :{0}".format(command2))

    command3 = 'sed -i "s#ext_name#' + ext_name + '#g" ' + "config_" + ws_data_path + ".py"

    sed2_p = subprocess.call(command3, shell=True)
    logging.info("sed2 config_*.py \n command :{0}".format(command3))
    # run ftest
    command =  "python RunBackgroundScripts.py --inputConfig " +   "config_" + ws_data_path + ".py" + " --mode fTestParallel" + " > " + log_name + " 2>&1"
    run_ftest_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run ftest \n command :{0}".format(command))
    # run cp data 
    command = "cp outdir_" + ext_name + "/*.root" +  " " + inputpath_name + ws_data_path +"/" + cp_name
    cp_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("end: {}".format(time.time()))
 
# ---------------------------- data card part --------------------------- #
def run_yields(ws_sig_path,ws_bkg_path, inputpath_name,log_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Datacard")
    # run Runyields
    command = "python RunYields.py --inputWSDirMap 2017=" +inputpath_name+ "ws_gghh_" + ws_sig_path+" --sigModelWSDir " + inputpath_name+ "ws_gghh_" + ws_sig_path+" --bkgModelWSDir " + inputpath_name+  ws_bkg_path + " --cats auto --procs auto --batch local --ext "+ ws_sig_path+ ">" + log_name+ " 2>&1"

    run_yields_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run yields \n command :{0}".format(command))
    logging.info("end: {}".format(time.time()))

def run_makeDatacard(ws_sig_path,output_card_name,log_name,channel):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Datacard")
    # run RunmakeDatacard
      
    command = "python makeDatacard.py --years 2017 --prune --ext " + ws_sig_path + " --output "+ output_card_name+  ">" + log_name+ " 2>&1"

    run_makeDatacard_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run makeDatacard \n command :{0}".format(command))
    # write the branching ratio info in it 
    if (channel=="FH") :
        add_br_note = open(output_card_name+".txt", 'a')
        add_br_note.write("CMS_wwgg_br_HH_WWgg      rateParam  *  gghh*  0.000970198 \nCMS_wwgg_br_WW_4Q     rateParam  *  gghh*  0.4489 \nnuisance  edit  freeze  CMS_wwgg_br_HH_WWgg \nnuisance  edit  freeze  CMS_wwgg_br_WW_4Q")
    elif(channel=="SL"):
        add_br_note = open(output_card_name+".txt", 'a')
        add_br_note.write("CMS_wwgg_br_HH_WWgg      rateParam  *  gghh*  0.000970198 \n CMS_wwgg_br_WW_2Qlnu     rateParam  *  gghh*  0.441 \n nuisance  edit  freeze  CMS_wwgg_br_HH_WWgg \n nuisance  edit  freeze  CMS_wwgg_br_WW_2Qlnu")
    elif(channel=="FHSL"):
        add_br_note = open(output_card_name+".txt", 'a')
        add_br_note.write("CMS_wwgg_br_HH_WWgg      rateParam  *  gghh*  0.000970198 \n CMS_wwgg_br_WW_4Q_2Qlnu     rateParam  *  gghh*  0.8899 \n nuisance  edit  freeze  CMS_wwgg_br_HH_WWgg \n nuisance  edit  freeze  CMS_wwgg_br_WW_4Q_2Qlnu")
    logging.info("end: {}".format(time.time()))

def run_combine(output_card_name,log_name,output_file_name):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Datacard")
    # run Runcombine
    command = "combine -M AsymptoticLimits -m 125 --rMax 500000 -n " + output_file_name+" "+ output_card_name + ".txt" + " --run expected " + ">" + log_name+  " 2>&1 "
    run_combine_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run combine \n command :{0}".format(command))
    logging.info("end: {}".format(time.time()))
def run_combine_card():
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Datacard")
    # combine the datacard for same masspoint with categories
    command = "combineCards.py" + "FHSL_1jets_M3000_cat0=Datacard_M3000_1jets_cat0_FHSL.txt FHSL_1jets_M3000_cat1=Datacard_M3000_1jets_cat1_FHSL.txt FHSL_1jets_M3000_cat2=Datacard_M3000_1jets_cat2_FHSL.txt FHSL_1jets_M3000_cat3=Datacard_M3000_1jets_cat3_FHSL.txt   > Datacard_combined_FHSL_1jets_M3000.txt"
    run_combine_card = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
if __name__ == "__main__":

    # 读取categories.json文件
    with codecs.open('categories_all_mass_point.json', 'r', encoding='utf-8') as f:
        categories = json.load(f)
    keys = list(categories.keys())
    # 创建一个字典来存储每个key的category names
    category_names = defaultdict(list)

    # 循环遍历categories中的每个key和value
    for key, value in categories.items():
        # 获取value的长度
        num_categories = len(value)
        # 创建一个包含相应数量的category names的列表
        cat_names = ['cat{}'.format(i) for i in range(num_categories)]
        # 将这个列表存储到category_names字典中
        category_names[key] = cat_names





# example
# category_name = list_category_name[index_category_name]
# cat = "FHSamples_SL_boosted_cat"
# cat_list = ['FHSamples_SL_boosted_cat','FHSamples_SL_fullyresovled_cat','FHSamples_SL_merged_boosted_cat','FHSamples_FH_2Wfatjet_cat','FHSamples_FH_1Wfatjet_cat','FHSamples_FH_fully_resovled_cat']
cat_list = ['FHSamples_SL_fullyresovled_cat']
for cat in cat_list:
    # ------------------------------------- background fit -------------------------------------
    input_path_name = "/eos/user/z/zhenxuan/hhwwgg_workspace/hhwwgg_root/hhwwgg_root_FH/M1100/"
    ws_data_path = "ws_2017_"+ cat
    output_data_root_name = "Data_2017_"+ cat + "_"+ "M1100" + ".root"
    log_name_data = "bkg_2017_"+cat+".log"
    run_Tree2WS_data(inputpath_name = input_path_name , ws_data_path=ws_data_path, output_data_root_name=output_data_root_name)
    ext_name = "ws_2017_" + cat
    run_backgroundfit(ws_data_path=ws_data_path, log_name = log_name_data , inputpath_name = input_path_name , ext_name=ext_name, cp_name="CMS-HGG_multipdf_RECO_untagged_"+cat+"_2017.root")
    # ------------------------------------- signal fit -------------------------------------

    input_file_name_signal = "Signal_"+ "M1100" +"_2017_"+cat+".root"
    log_file_name_signal = "M1100" +"_hhwwgg_MC_2017_"+cat+".log"
    ws_path_signal = "M1100" +"_2017_"+cat
    output_root_name_signal = "output_Signal"+"M1100" + cat + "_M125_2017_13TeV_amcatnloFXFX_pythia8_gghh.root"
    run_Tree2WS_sig(inputpath_name = input_path_name ,inputfile_name=input_file_name_signal,log_file_name= log_file_name_signal, ws_path = ws_path_signal, output_sig_root_name=output_root_name_signal)
    run_ftest(ws_path = ws_path_signal, log_name = "signal_ftest_" + log_file_name_signal, inputpath_name= input_path_name)
    run_signalfit(ws_path = ws_path_signal, log_name = "signal_signalfit_" + log_file_name_signal, inputpath_name=input_path_name)
    run_signal_plot(cats="RECO_untagged_"+cat, exts="dcb_2017_" + "M1100_2017_" + cat, outputExt="packaged_" + "M1100_2017_" + cat, log_packaged_name = "packaged_" + "M1100" + cat+".log", ws_path=ws_path_signal, inputpath_name =input_path_name, log_plotter_name = "plotter_" + "M1100" + cat+".log", cp_name="CMS-HGG_sigfit_packaged_RECO_untagged_"+cat+"_2017.root")
    run_yields(ws_sig_path=ws_path_signal, log_name = "M1100_2017_" + cat +"_yields.log" , inputpath_name = input_path_name, ws_bkg_path = ws_data_path )
    run_makeDatacard(ws_sig_path=ws_path_signal, log_name = "M1100" + cat + "_makeDatacard.log" , output_card_name = "Datacard_" + "M1100_2017_" + cat, channel="FH" )
    run_combine(output_file_name="M1100_2017_" + cat, log_name = "M1100_2017_" + cat +"_combine_limit.log" , output_card_name = "Datacard_" + "M1100_2017_" + cat)

# # # ----------------------------- 1jet cat all sig ----------------------------- #
# # # attention: have to put the signal root file in ws_gghh
# run_Tree2WS_sig(inputpath_name = input_path_name + list_mass[index_mass] + "/",inputfile_name="Signal_"+list_mass[index_mass]+"_FHSL_2017_1jets_"+category_name+".root",log_file_name= list_mass[index_mass]+"_hhwwgg_MC_FHSL_1jets_"+category_name+".log", ws_path = list_mass[index_mass]+"_1jets_"+category_name, output_sig_root_name="output_Signal"+list_mass[index_mass]+"_1jets_"+category_name+"_M125_FHSL_2017_13TeV_amcatnloFXFX_pythia8_gghh.root")
# run_ftest(ws_path = list_mass[index_mass] + "_1jets_"+category_name, log_name = "signal_ftest_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL.log", inputpath_name= input_path_name + list_mass[index_mass] + "/")
# run_signalfit(ws_path = list_mass[index_mass] + "_1jets_"+category_name, log_name = "signal_signalfit_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL.log", inputpath_name=input_path_name + list_mass[index_mass] + "/")
# run_signal_plot(cats="RECO_untagged_1jets_"+category_name, exts="dcb_2017_" + list_mass[index_mass] + "_1jets_"+category_name, outputExt="packaged_" + list_mass[index_mass] + "_1jets_"+category_name, log_packaged_name = "packaged_" + list_mass[index_mass] + "_1jets_"+category_name+".log", ws_path=list_mass[index_mass] + "_1jets_"+category_name, inputpath_name =input_path_name + list_mass[index_mass] + "/", log_plotter_name = "plotter_" + list_mass[index_mass] + "_1jets_"+category_name+".log", cp_name="CMS-HGG_sigfit_packaged_RECO_untagged_1jets_"+category_name+"_2017.root")
# run_yields(ws_sig_path=list_mass[index_mass] + "_1jets_"+category_name, log_name = list_mass[index_mass] + "_FH_1jets_"+category_name+"_yields.log" , inputpath_name = input_path_name + list_mass[index_mass] + "/", ws_bkg_path = "ws_1jets_"+category_name )
# run_makeDatacard(ws_sig_path=list_mass[index_mass] + "_1jets_"+category_name, log_name = list_mass[index_mass] + "_FH_1jets_"+category_name+"_makeDatacard.log" , output_card_name = "Datacard_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL", channel="FHSL" )
# run_combine(output_file_name=list_mass[index_mass] + "_1jets_"+category_name, log_name = list_mass[index_mass] + "_FHSL_1jets_"+category_name+"_combine_limit.log" , output_card_name = "Datacard_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL")
#---------#
    # 现在你可以按需要使用category_names字典来访问每个key的category names
    # 例如，要访问'M250'的category names：
    # print(category_names['M250'])
    # list_mass = keys
    # for index_mass in range(len(list_mass)):
    #     list_category_name = category_names[list_mass[index_mass]]
    #     for index_category_name in range(len(list_category_name)):
    #         category_name = list_category_name[index_category_name]
            # run_Tree2WS_data(inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/", ws_data_path="ws_1jets_"+category_name, output_data_root_name="Data_FHSL_2017_cat_1jets_"+category_name+ "_"+ list_mass[index_mass] + ".root")
            # run_backgroundfit(ws_data_path="ws_1jets_"+category_name, log_name = "bkg_ws_1jets_"+category_name+".log" , inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/", ext_name="FHSL_ws_1jets_"+category_name, cp_name="CMS-HGG_multipdf_RECO_untagged_1jets_"+category_name+"_2017.root")
    #         # # ----------------------------- 1jet cat all sig ----------------------------- #
    #         # # attention: have to put the signal root file in ws_gghh
    #         run_Tree2WS_sig(inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/",inputfile_name="Signal_"+list_mass[index_mass]+"_FHSL_2017_1jets_"+category_name+".root",log_file_name= list_mass[index_mass]+"_hhwwgg_MC_FHSL_1jets_"+category_name+".log", ws_path = list_mass[index_mass]+"_1jets_"+category_name, output_sig_root_name="output_Signal"+list_mass[index_mass]+"_1jets_"+category_name+"_M125_FHSL_2017_13TeV_amcatnloFXFX_pythia8_gghh.root")
    #         run_ftest(ws_path = list_mass[index_mass] + "_1jets_"+category_name, log_name = "signal_ftest_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL.log", inputpath_name= "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/")
    #         run_signalfit(ws_path = list_mass[index_mass] + "_1jets_"+category_name, log_name = "signal_signalfit_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL.log", inputpath_name="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/")
    #         run_signal_plot(cats="RECO_untagged_1jets_"+category_name, exts="dcb_2017_" + list_mass[index_mass] + "_1jets_"+category_name, outputExt="packaged_" + list_mass[index_mass] + "_1jets_"+category_name, log_packaged_name = "packaged_" + list_mass[index_mass] + "_1jets_"+category_name+".log", ws_path=list_mass[index_mass] + "_1jets_"+category_name, inputpath_name ="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/", log_plotter_name = "plotter_" + list_mass[index_mass] + "_1jets_"+category_name+".log", cp_name="CMS-HGG_sigfit_packaged_RECO_untagged_1jets_"+category_name+"_2017.root")
    #         run_yields(ws_sig_path=list_mass[index_mass] + "_1jets_"+category_name, log_name = list_mass[index_mass] + "_FH_1jets_"+category_name+"_yields.log" , inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/" + list_mass[index_mass] + "/", ws_bkg_path = "ws_1jets_"+category_name )
    #         run_makeDatacard(ws_sig_path=list_mass[index_mass] + "_1jets_"+category_name, log_name = list_mass[index_mass] + "_FH_1jets_"+category_name+"_makeDatacard.log" , output_card_name = "Datacard_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL", channel="FHSL" )
    #         run_combine(output_file_name=list_mass[index_mass] + "_1jets_"+category_name, log_name = list_mass[index_mass] + "_FHSL_1jets_"+category_name+"_combine_limit.log" , output_card_name = "Datacard_" + list_mass[index_mass] + "_1jets_"+category_name+"_FHSL")
        # run_combine_card()