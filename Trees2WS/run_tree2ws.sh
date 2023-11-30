WhichSamples=${1}
if [ ${WhichSamples} -eq 0 ]
  then
    echo "running data"
    # new version with PN taggers
    python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/M250/Data_FHSL_2017_cat_1jets_cat0_M250.root
    # python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/Data_FH_2017_cat_1jet_s200.root
    # python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/Data_FH_2017_cat_4jets.root


    ##########

    # python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Data_FH_2017_cat_1jet.root
    # python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Data_FH_2017_cat_2jets_3jets.root
    # python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Data_FH_2017_cat_4jets.root

    #############
fi
if [ ${WhichSamples} -eq 1 ]
  then
    echo "running SL channel"
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FHSL_custom/M250/  --inputMass 125 --productionMode gghh --year 2017
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M300_SL_2017_2jets.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_2jets_SL.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M500_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_SL.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M1000_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_SL.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M2000_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_SL.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M3000_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_SL.log 2>&1

fi
if [ ${WhichSamples} -eq 2 ]
  then
    echo "running FH channel all cat"
    # new verision with PN tagger:
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/Signal_M1700_SL_2017_2jets_muon_unpassed100cut.root --inputMass 125 --productionMode gghh  --year 2017  
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/Signal_M3000_FH_2017_1jet_s200.root --inputMass 125 --productionMode gghh  --year 2017  >M3000_hhwwgg_MC_FH_1jet_s200.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/Signal_M3000_FH_2017_4jets.root --inputMass 125 --productionMode gghh  --year 2017  >M3000_hhwwgg_MC_FH_4jets.log 2>&1


    # ###########

    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile //eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M300_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_FH_1jet.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M500_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_FH_1jet.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M700_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M700_hhwwgg_MC_FH_1jet.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1000_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_FH_1jet.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1500_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M1500_hhwwgg_MC_FH_1jet.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2000_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_FH_1jet.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2600_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M2600_hhwwgg_MC_FH_1jet.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M3000_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_FH_1jet.log 2>&1

    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M300_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M500_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M700_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M700_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1000_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1500_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M1500_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2000_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_FH_2jets_3jets.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2600_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M2600_hhwwgg_MC_FH_2jets_3jets.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M3000_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_FH_2jets_3jets.log 2>&1

    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M300_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_FH_4jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M500_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_FH_4jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M700_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M700_hhwwgg_MC_FH_4jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1000_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_FH_4jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1500_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M1500_hhwwgg_MC_FH_4jets.log 2>&1 
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2000_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_FH_4jets.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2600_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M2600_hhwwgg_MC_FH_4jets.log 2>&1
    # python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M3000_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_FH_4jets.log 2>&1

   

fi
if [ ${WhichSamples} -eq 3 ]
  then
    echo "cp to different category"
    # ------------------------ new verision with PN tagger ----------------------- #
    command = "cp " + inputpath_name +"ws_gghh/" + inputfile_name.split('.root')[0] + '_gghh' + ".root" + " " + inputpath_name +"ws_gghh_"+ws_path +"/" + output_sig_root_name
    run_Tree2WS_sig(inputpath_name = "/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_ForSW_SL_resolved/",inputfile_name="Signal_"+"M1700"+"_SL_2017_2jets_"+category_name+".root",log_file_name= "M1700"+"_hhwwgg_MC_2jets_SL_"+category_name+".log", ws_path = "M1700"+"_2jets_SL_"+category_name, output_sig_root_name="output_Signal"+"M1700"+"_2jets_SL_"+category_name+"_M125_SL_2017_13TeV_amcatnloFXFX_pythia8_gghh.root")

    # outputpath="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_1jet_l200"
    # echo $outputpath
    # mkdir $outputpath
    # cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh/Signal_M3000_FH_2017_1jet_l200_gghh.root $outputpath/output_Signal3000_M125_FH_2017_13TeV_amcatnloFXFX_pythia8_gghh_1jet_l200.root
    
    # outputpath="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_1jet_s200"
    # echo $outputpath
    # mkdir $outputpath
    # cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh/Signal_M3000_FH_2017_1jet_s200_gghh.root $outputpath/output_Signal3000_M125_FH_2017_13TeV_amcatnloFXFX_pythia8_gghh_1jet_s200.root
    
    # outputpath="/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_4jets"
    # echo $outputpath
    # mkdir $outputpath
    # cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh/Signal_M3000_FH_2017_4jets_gghh.root $outputpath/output_Signal3000_M125_FH_2017_13TeV_amcatnloFXFX_pythia8_gghh_4jets.root
    
    # cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh/Signal_M3000_FH_2017_1jet_s200_gghh.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_1jet_s200/output_Signal3000_M125_FH_2017_13TeV_amcatnloFXFX_pythia8_gghh_1jet_s200.root
    # cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh/Signal_M3000_FH_2017_4jets_gghh.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_4jets/output_Signal3000_M125_FH_2017_13TeV_amcatnloFXFX_pythia8_gghh_4jets.root

    # ------------------------ new verision with PN tagger ----------------------- #

    # cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/ws_gghh/Signal_M300_SL_2017_2jets_gghh.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/ws_gghh_M300_SL/output_Signal300_M125_SL_2017_13TeV_amcatnloFXFX_pythia8_gghh_2jets.root

fi