'''
Hi my friend, believe it or not, I just want to create a king script to create limit with flashggfinal fit framework from ntuples(root) with only one script
let see if the magic can work
'''
import uproot 
import awkward as ak
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

def run_combine_limit(output_card_name,log_name,output_file_name, rMax):
    logging.info("begin: {}".format(time.time()))
    os.chdir("/afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Datacard")
    # run Runcombine
    if rMax != False:
        command = "combine -M AsymptoticLimits --rMax 500000 -m 125 -n " + output_file_name+" "+ output_card_name + ".txt" + " --run expected " + ">" + log_name+  " 2>&1 "
    else:
        command = "combine -M AsymptoticLimits -m 125 -n " + output_file_name+" "+ output_card_name + ".txt" + " --run expected " + ">" + log_name+  " 2>&1 "

    run_combine_p = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    logging.info("run combine \n command :{0}".format(command))
    logging.info("end: {}".format(time.time()))


    
if __name__ == "__main__":
    list_category = ["electron_unpassed100cut", "electron_passed100cut","muon_unpassed100cut", "muon_passed100cut"]
    # list_category = ["electron_unpassed100cut"]
    # Data
    # for i in range(len(list_category)):
    #     category_name = list_category[i]
    #     run_backgroundfit(ws_data_path="ws_2jets_SL_"+category_name, log_name = "bkg_ws_2jets_SL_"+category_name+".log" , inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/", ext_name="ws_2jets_SL_"+category_name, cp_name="CMS-HGG_multipdf_RECO_untagged_2jets_SL_"+category_name+"_2017.root")
    #     run_Tree2WS_data(inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/", ws_data_path="ws_2jets_SL_"+category_name, output_data_root_name="Data_2017_SL_2jets_"+category_name+".root")
    # ------------------------------------- - ------------------------------------ #
    # signal mass
    list_mass = ["M250", "M270", "M300", "M320", "M350", "M400", "M450", "M550", "M600", "M650", "M700", "M750", "M800", "M850", "M900", "M1000", "M1100", "M1200", "M1300", "M1400", "M1500", "M1600", "M1700", "M1800", "M1900", "M2000", "M2400", "M2600", "M2600", "M2700", "M2800", "M3000"]
    # list_mass = ["M250"]
    for index_category in range(len(list_category)):
        category_name = list_category[index_category]
        for index_mass in range(len(list_mass)):
            mass = list_mass[index_mass]
            # run_Tree2WS_sig(inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/",inputfile_name="Signal_"+mass+"_SL_2017_2jets_"+category_name+".root",log_file_name= mass+"_hhwwgg_MC_2jets_SL_"+category_name+".log", ws_path = mass+"_2jets_SL_"+category_name, output_sig_root_name="output_Signal"+mass+"_2jets_SL_"+category_name+"_M125_SL_2017_13TeV_amcatnloFXFX_pythia8_gghh.root")
            # run_ftest(ws_path = mass + "_2jets_SL_"+category_name, log_name = "signal_ftest_" + mass + "_2jets_SL_"+category_name+"_SL.log", inputpath_name= "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/")
            # run_signalfit(ws_path = mass + "_2jets_SL_"+category_name, log_name = "signal_signalfit_" + mass + "_2jets_SL_"+category_name+"_SL.log", inputpath_name="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/")
            # run_signal_plot(cats="RECO_untagged_2jets_SL_"+category_name, exts="dcb_2017_" + mass + "_2jets_SL_"+category_name, outputExt="packaged_" + mass + "_2jets_SL_"+category_name, log_packaged_name = "packaged_" + mass + "_2jets_SL_"+category_name+".log", ws_path=mass + "_2jets_SL_"+category_name, inputpath_name ="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/", log_plotter_name = "plotter_" + mass + "_2jets_SL_"+category_name+".log", cp_name="CMS-HGG_sigfit_packaged_RECO_untagged_2jets_SL_"+category_name+"_2017.root")
            run_yields(ws_sig_path=mass + "_2jets_SL_"+category_name, log_name = mass + "_2jets_SL_"+category_name+"_yields.log" , inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/", ws_bkg_path = "ws_2jets_SL_"+category_name )
            run_makeDatacard(ws_sig_path=mass + "_2jets_SL_"+category_name, log_name = mass + "_2jets_SL_"+category_name+"_makeDatacard.log" , output_card_name = "Datacard_" + mass + "_2jets_SL_"+category_name+"_SL", channel="SL" )
            run_combine_limit(output_file_name=mass + "_2jets_SL_"+category_name, log_name = mass + "_SL_2jets_SL_"+category_name+"_makeDatacard.log" , output_card_name = "Datacard_" + mass + "_2jets_SL_"+category_name+"_SL", rMax = 100000) # if rMax== False, mean we don't add rMax option
            # run_combine_datacard()