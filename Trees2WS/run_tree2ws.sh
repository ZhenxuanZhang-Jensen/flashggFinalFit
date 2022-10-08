WhichSamples=${1}
if [ ${WhichSamples} -eq 0 ]
  then
    echo "running data"
    python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Data_FH_2017_cat_1jet.root
    python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Data_FH_2017_cat_2jets_3jets.root
    python trees2ws_data.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Data_FH_2017_cat_4jets.root
fi
if [ ${WhichSamples} -eq 1 ]
  then
    echo "running SL channel"
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M500_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_SL.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M1000_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_SL.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M2000_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_SL.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/Signal_M3000_SL_2017.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_SL.log 2>&1

fi
if [ ${WhichSamples} -eq 2 ]
  then
    echo "running FH channel all cat"
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M300_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_FH_1jet.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M500_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_FH_1jet.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M700_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M700_hhwwgg_MC_FH_1jet.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1000_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_FH_1jet.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1500_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M1500_hhwwgg_MC_FH_1jet.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2000_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_FH_1jet.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2600_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M2600_hhwwgg_MC_FH_1jet.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M3000_FH_2017_1jet.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_FH_1jet.log 2>&1

    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M300_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M500_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M700_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M700_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1000_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1500_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M1500_hhwwgg_MC_FH_2jets_3jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2000_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_FH_2jets_3jets.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2600_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M2600_hhwwgg_MC_FH_2jets_3jets.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M3000_FH_2017_2jets_3jets.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_FH_2jets_3jets.log 2>&1

    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M300_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M300_hhwwgg_MC_FH_4jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M500_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M500_hhwwgg_MC_FH_4jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M700_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M700_hhwwgg_MC_FH_4jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1000_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M1000_hhwwgg_MC_FH_4jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M1500_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M1500_hhwwgg_MC_FH_4jets.log 2>&1 
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2000_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M2000_hhwwgg_MC_FH_4jets.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M2600_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M2600_hhwwgg_MC_FH_4jets.log 2>&1
    python trees2ws.py --inputConfig config_simple.py --inputTreeFile /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/Signal_M3000_FH_2017_4jets.root --inputMass 125 --productionMode gghh --year 2017  >M3000_hhwwgg_MC_FH_4jets.log 2>&1

   

fi
if [ ${WhichSamples} -eq 3 ]
  then
    echo "cp to different category"
    cp ws_gghh/ ws_gghh_M300/output_Signal300_M125_FH_2017_13TeV_amcatnloFXFX_pythia8_gghh.root

fi