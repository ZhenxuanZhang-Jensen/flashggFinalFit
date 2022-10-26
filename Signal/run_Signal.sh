WhichSamples=${1}
if [ ${WhichSamples} -eq 0 ]
  then
    echo "running FH signal fTest"
    # already saved different mass point root file into different directory
    # can't run parallel since some operation may have confliction attention
    
    python RunSignalScripts.py --inputConfig config_M300_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M300_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M300_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M300_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M300_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M300_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M500_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M500_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M500_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M500_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M500_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M500_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M700_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M700_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M700_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M700_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M700_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M700_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M1000_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M1000_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1000_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M1000_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1000_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M1000_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M1500_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M1500_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1500_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M1500_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1500_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M1500_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M2000_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M2000_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2000_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M2000_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2000_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M2000_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M2600_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M2600_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2600_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M2600_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2600_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M2600_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M3000_1jet_FH_2017.py --mode 'fTest' >signal_ftest_M3000_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M3000_2jets_3jets_FH_2017.py --mode 'fTest' >signal_ftest_M3000_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M3000_4jets_FH_2017.py --mode 'fTest' >signal_ftest_M3000_4jets_FH.log 2>&1
fi
if [ ${WhichSamples} -eq 10 ]
  then
    echo "running SL signal fTest"
    # already saved different mass point root file into different directory
    # can't run parallel since some operation may have confliction attention
    
    python RunSignalScripts.py --inputConfig config_M300_2jets_SL_2017.py --mode 'fTest' >signal_ftest_M300_2jets_SL.log 2>&1 
    # python RunSignalScripts.py --inputConfig config_M300_2jets_3jets_SL_2017.py --mode 'fTest' >signal_ftest_M300_2jets_3jets_SL.log 2>&1 
    # python RunSignalScripts.py --inputConfig config_M300_4jets_SL_2017.py --mode 'fTest' >signal_ftest_M300_4jets_SL.log 2>&1

fi
if [ ${WhichSamples} -eq 11 ]
  then
    echo "running SL signal fit"
    # already saved different mass point root file into different directory
    # can't run parallel since some operation may have confliction attention
    python RunSignalScripts.py --inputConfig config_M300_2jets_SL_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalfit_M300_2jets_SL.log 2>&1 

fi
if [ ${WhichSamples} -eq 12 ]
  then
  #attention: need to copy all the CMS_multipdf* to your workspace dir
    echo 'run copy'
    
    cp outdir_dcb_2017_M300_2jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M300_2jets_gghh_2017_RECO_untagged_2jets_SL.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/ws_gghh_M300_SL_2jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_SL_2017.root
fi
if [ ${WhichSamples} -eq 13 ]
  then
    echo 'run packaged'
    # already saved different mass point root file into different directory
    python RunPackager.py --cats RECO_untagged_2jets_SL --exts dcb_2017_M300_2jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M300_2jets_SL
fi
if [ ${WhichSamples} -eq 14 ]
  then
    echo 'run plotter'
    # already saved different mass point root file into different directory
    python RunPlotter.py --cats RECO_untagged_2jets_SL --procs all --years 2017 --ext packaged_M300_2jets_SL

fi



if [ ${WhichSamples} -eq 1 ]
  then
    echo ${whichsample}
    # already saved different mass point root file into different directory
    # can't run parallel since some operation may have confliction attention
    python RunSignalScripts.py --inputConfig config_M300_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M300_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M300_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M300_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M300_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M300_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M500_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M500_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M500_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M500_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M500_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M500_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M700_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M700_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M700_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M700_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M700_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M700_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M1000_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M1000_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1000_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M1000_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1000_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M1000_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M1500_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M1500_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1500_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M1500_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1500_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M1500_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M2000_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M2000_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2000_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M2000_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2000_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M2000_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M2600_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M2600_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2600_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M2600_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2600_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M2600_4jets_FH.log 2>&1

    python RunSignalScripts.py --inputConfig config_M3000_1jet_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M3000_1jet_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M3000_2jets_3jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M3000_2jets_3jets_FH.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M3000_4jets_FH_2017.py --mode 'signalFit' --modeOpts "--skipSystematics" >signal_signalFit_M3000_4jets_FH.log 2>&1

fi
if [ ${WhichSamples} -eq 2 ]
  then
    echo 'run doPlots'
    # already saved different mass point root file into different directory
    python RunSignalScripts.py --inputConfig config_M300_2017.py --mode 'signalFit' --modeOpts "--skipSystematics --doPlots --skipVertexScenarioSplit" >signal_signalFit_M300.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M700_2017.py --mode 'signalFit' --modeOpts "--skipSystematics --doPlots --skipVertexScenarioSplit" >signal_signalFit_M700.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M1250_2017.py --mode 'signalFit' --modeOpts "--skipSystematics --doPlots --skipVertexScenarioSplit" >signal_signalFit_M1250.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M2000_2017.py --mode 'signalFit' --modeOpts "--skipSystematics --doPlots --skipVertexScenarioSplit" >signal_signalFit_M2000.log 2>&1 
    python RunSignalScripts.py --inputConfig config_M3000_2017.py --mode 'signalFit' --modeOpts "--skipSystematics --doPlots --skipVertexScenarioSplit" >signal_signalFit_M3000.log 2>&1 
fi
if [ ${WhichSamples} -eq 3 ]
  then
    echo 'run packaged'
    # already saved different mass point root file into different directory
    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M300_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M300_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M300_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M300_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M300_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M300_4jets
    
    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M500_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M500_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M500_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M500_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M500_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M500_4jets

    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M700_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M700_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M700_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M700_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M700_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M700_4jets

    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M1000_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M1000_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M1000_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M1000_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M1000_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M1000_4jets

    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M1500_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M1500_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M1500_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M1500_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M1500_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M1500_4jets

    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M2000_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M2000_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M2000_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M2000_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M2000_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M2000_4jets

    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M2600_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M2600_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M2600_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M2600_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M2600_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M2600_4jets

    python RunPackager.py --cats RECO_untagged_1jet --exts dcb_2017_M3000_1jet,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M3000_1jet
    python RunPackager.py --cats RECO_untagged_2jets_3jets --exts dcb_2017_M3000_2jets_3jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M3000_2jets_3jets
    python RunPackager.py --cats RECO_untagged_4jets --exts dcb_2017_M3000_4jets,  --batch local  --massPoints 125 --year 2017 --outputExt packaged_M3000_4jets

fi
if [ ${WhichSamples} -eq 4 ]
  then
    echo 'run plotter'
    # already saved different mass point root file into different directory
    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M300_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M300_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M300_4jets
    
    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M500_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M500_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M500_4jets

    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M700_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M700_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M700_4jets

    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M1000_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M1000_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M1000_4jets

    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M1500_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M1500_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M1500_4jets

    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M2000_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M2000_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M2000_4jets

    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M2600_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M2600_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M2600_4jets

    python RunPlotter.py --cats RECO_untagged_1jet --procs all --years 2017 --ext packaged_M3000_1jet
    python RunPlotter.py --cats RECO_untagged_2jets_3jets --procs all --years 2017 --ext packaged_M3000_2jets_3jets
    python RunPlotter.py --cats RECO_untagged_4jets --procs all --years 2017 --ext packaged_M3000_4jets

fi

if [ ${WhichSamples} -eq 5 ]
  then
  #attention: need to copy all the CMS_multipdf* to your workspace dir
    echo 'run copy'
    
    cp outdir_dcb_2017_M300_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M300_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M500_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M500_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M700_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M700_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M1000_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M1000_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M1500_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M1500_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M2000_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M2000_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M2600_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M2600_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root
    cp outdir_dcb_2017_M3000_1jet/signalFit/output/CMS-HGG_sigfit_dcb_2017_M3000_1jet_gghh_2017_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_1jet/CMS-HGG_sigfit_packaged_RECO_untagged_1jet_2017.root

    cp outdir_dcb_2017_M300_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M300_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M500_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M500_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M700_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M700_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M1000_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M1000_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M1500_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M1500_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M2000_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M2000_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M2600_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M2600_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root
    cp outdir_dcb_2017_M3000_2jets_3jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M3000_2jets_3jets_gghh_2017_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_2jets_3jets/CMS-HGG_sigfit_packaged_RECO_untagged_2jets_3jets_2017.root

    cp outdir_dcb_2017_M300_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M300_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M500_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M500_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M700_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M700_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M1000_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M1000_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M1500_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M1500_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M2000_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M2000_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M2600_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M2600_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
    cp outdir_dcb_2017_M3000_4jets/signalFit/output/CMS-HGG_sigfit_dcb_2017_M3000_4jets_gghh_2017_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_4jets/CMS-HGG_sigfit_packaged_RECO_untagged_4jets_2017.root
fi

