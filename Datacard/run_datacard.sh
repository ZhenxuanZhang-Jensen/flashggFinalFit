WhichSamples=${1}
if [ ${WhichSamples} -eq 10 ]
  then
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/ws_gghh_M300_SL_2jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/ws_gghh_M300_SL_2jets --bkgModelWSDir  /eos/user/s/shsong/hhwwgg_root/hhwwgg_root_SL/ws_2jets --cats auto --procs auto --batch local --ext M300_2jets_SL >M300_SL_2jets_SL_yields.log 2>&1
   
fi
if [ ${WhichSamples} -eq 0 ]
  then
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M300_1jet >M300_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M500_1jet >M500_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M700_1jet >M700_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M1000_1jet >M1000_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M1500_1jet >M1500_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M2000_1jet >M2000_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M2600_1jet >M2600_FH_1jet_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_1jet --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_1jet --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet --cats auto --procs auto --batch local --ext M3000_1jet >M3000_FH_1jet_yields.log 2>&1

    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M300_2jets_3jets >M300_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M500_2jets_3jets >M500_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M700_2jets_3jets >M700_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M1000_2jets_3jets >M1000_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M1500_2jets_3jets >M1500_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M2000_2jets_3jets >M2000_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M2600_2jets_3jets >M2600_FH_2jets_3jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_2jets_3jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_2jets_3jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets --cats auto --procs auto --batch local --ext M3000_2jets_3jets >M3000_FH_2jets_3jets_yields.log 2>&1

    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M300_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M300_4jets >M300_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M500_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M500_4jets >M500_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M700_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M700_4jets >M700_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1000_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M1000_4jets >M1000_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M1500_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M1500_4jets >M1500_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2000_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M2000_4jets >M2000_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M2600_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M2600_4jets >M2600_FH_4jets_yields.log 2>&1
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_4jets --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_gghh_M3000_4jets --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets --cats auto --procs auto --batch local --ext M3000_4jets >M3000_FH_4jets_yields.log 2>&1
   
fi
if [ ${WhichSamples} -eq 11 ]
  then
    python makeDatacard.py --years 2017 --prune --ext 'M300_2jets_SL' --output Datacard_M300_2jets_SL_SL
    
fi
if [ ${WhichSamples} -eq 1 ]
  then
    python makeDatacard.py --years 2017 --prune --ext 'M300_1jet' --output Datacard_M300_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M300_2jets_3jets' --output Datacard_M300_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M300_4jets' --output Datacard_M300_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M500_1jet' --output Datacard_M500_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M500_2jets_3jets' --output Datacard_M500_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M500_4jets' --output Datacard_M500_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M700_1jet' --output Datacard_M700_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M700_2jets_3jets' --output Datacard_M700_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M700_4jets' --output Datacard_M700_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M1000_1jet' --output Datacard_M1000_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M1000_2jets_3jets' --output Datacard_M1000_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M1000_4jets' --output Datacard_M1000_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M1500_1jet' --output Datacard_M1500_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M1500_2jets_3jets' --output Datacard_M1500_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M1500_4jets' --output Datacard_M1500_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M2000_1jet' --output Datacard_M2000_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M2000_2jets_3jets' --output Datacard_M2000_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M2000_4jets' --output Datacard_M2000_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M2600_1jet' --output Datacard_M2600_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M2600_2jets_3jets' --output Datacard_M2600_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M2600_4jets' --output Datacard_M2600_4jets_FH
    
    python makeDatacard.py --years 2017 --prune --ext 'M3000_1jet' --output Datacard_M3000_1jet_FH
    python makeDatacard.py --years 2017 --prune --ext 'M3000_2jets_3jets' --output Datacard_M3000_2jets_3jets_FH
    python makeDatacard.py --years 2017 --prune --ext 'M3000_4jets' --output Datacard_M3000_4jets_FH
    
fi
if [ ${WhichSamples} -eq 2 ]
  then
    python cleanDatacard.py --datacard Datacard_M500_SL.txt --factor 2 --removeDoubleSided
    python cleanDatacard.py --datacard Datacard_M1000_SL.txt --factor 2 --removeDoubleSided
    python cleanDatacard.py --datacard Datacard_M2000_SL.txt --factor 2 --removeDoubleSided
    python cleanDatacard.py --datacard Datacard_M3000_SL.txt --factor 2 --removeDoubleSided
    
fi
if [ ${WhichSamples} -eq 3 ]
  then
    # combineCards.py SL_M500=Datacard_combined_M500_SL.txt FH_M500=Datacard_combined_M500_FH.txt  > Datacard_combined_M500_SL_FH.txt
    # combine SL
    # combineCards.py SL_1jet_M300=Datacard_M300_1jet_SL.txt SL_2jets_M300=Datacard_M300_2jets_SL.txt  > Datacard_combined_M300_SL.txt
    # combineCards.py SL_1jet_M500=Datacard_M500_1jet_SL.txt SL_2jets_M500=Datacard_M500_2jets_SL.txt  > Datacard_combined_M500_SL.txt
    # combineCards.py SL_1jet_M700=Datacard_M700_1jet_SL.txt SL_2jets_M700=Datacard_M700_2jets_SL.txt  > Datacard_combined_M700_SL.txt
    # combineCards.py SL_1jet_M1000=Datacard_M1000_1jet_SL.txt SL_2jets_M1000=Datacard_M1000_2jets_SL.txt  > Datacard_combined_M1000_SL.txt
    # combineCards.py SL_1jet_M1500=Datacard_M1500_1jet_SL.txt SL_2jets_M1500=Datacard_M1500_2jets_SL.txt  > Datacard_combined_M1500_SL.txt
    # combineCards.py SL_1jet_M2000=Datacard_M2000_1jet_SL.txt SL_2jets_M2000=Datacard_M2000_2jets_SL.txt  > Datacard_combined_M2000_SL.txt
    # combineCards.py SL_1jet_M2600=Datacard_M2600_1jet_SL.txt SL_2jets_M2600=Datacard_M2600_2jets_SL.txt  > Datacard_combined_M2600_SL.txt
    # combineCards.py SL_1jet_M3000=Datacard_M3000_1jet_SL.txt SL_2jets_M3000=Datacard_M3000_2jets_SL.txt  > Datacard_combined_M3000_SL.txt
    # combine FH
    # combineCards.py FH_1jet_M300=Datacard_M300_1jet_FH.txt FH_2jets_3jets_M300=Datacard_M300_2jets_3jets_FH.txt FH_4jets_M300=Datacard_M300_4jets_FH.txt  > Datacard_combined_M300_FH.txt
    # combineCards.py FH_1jet_M500=Datacard_M500_1jet_FH.txt FH_2jets_3jets_M500=Datacard_M500_2jets_3jets_FH.txt FH_4jets_M500=Datacard_M500_4jets_FH.txt  > Datacard_combined_M500_FH.txt
    # combineCards.py FH_1jet_M700=Datacard_M700_1jet_FH.txt FH_2jets_3jets_M700=Datacard_M700_2jets_3jets_FH.txt FH_4jets_M700=Datacard_M700_4jets_FH.txt  > Datacard_combined_M700_FH.txt
    # combineCards.py FH_1jet_M1000=Datacard_M1000_1jet_FH.txt FH_2jets_3jets_M1000=Datacard_M1000_2jets_3jets_FH.txt FH_4jets_M1000=Datacard_M1000_4jets_FH.txt  > Datacard_combined_M1000_FH.txt
    # combineCards.py FH_1jet_M1500=Datacard_M1500_1jet_FH.txt FH_2jets_3jets_M1500=Datacard_M1500_2jets_3jets_FH.txt FH_4jets_M1500=Datacard_M1500_4jets_FH.txt  > Datacard_combined_M1500_FH.txt
    # combineCards.py FH_1jet_M2000=Datacard_M2000_1jet_FH.txt FH_2jets_3jets_M2000=Datacard_M2000_2jets_3jets_FH.txt FH_4jets_M2000=Datacard_M2000_4jets_FH.txt  > Datacard_combined_M2000_FH.txt
    # combineCards.py FH_1jet_M2600=Datacard_M2600_1jet_FH.txt FH_2jets_3jets_M2600=Datacard_M2600_2jets_3jets_FH.txt FH_4jets_M2600=Datacard_M2600_4jets_FH.txt  > Datacard_combined_M2600_FH.txt
    # combineCards.py FH_1jet_M3000=Datacard_M3000_1jet_FH.txt FH_2jets_3jets_M3000=Datacard_M3000_2jets_3jets_FH.txt FH_4jets_M3000=Datacard_M3000_4jets_FH.txt  > Datacard_combined_M3000_FH.txt
    # combine all
    # combineCards.py SL_M300=Datacard_combined_M300_SL.txt FH_M300=Datacard_combined_M300_FH.txt  > Datacard_combined_M300_SL_FH.txt
    # combineCards.py SL_M500=Datacard_combined_M500_SL.txt FH_M500=Datacard_combined_M500_FH.txt  > Datacard_combined_M500_SL_FH.txt
    # combineCards.py SL_M700=Datacard_combined_M700_SL.txt FH_M700=Datacard_combined_M700_FH.txt  > Datacard_combined_M700_SL_FH.txt
    # combineCards.py SL_M1000=Datacard_combined_M1000_SL.txt FH_M1000=Datacard_combined_M1000_FH.txt  > Datacard_combined_M1000_SL_FH.txt
    # combineCards.py SL_M1500=Datacard_combined_M1500_SL.txt FH_M1500=Datacard_combined_M1500_FH.txt  > Datacard_combined_M1500_SL_FH.txt
    # combineCards.py SL_M2000=Datacard_combined_M2000_SL.txt FH_M2000=Datacard_combined_M2000_FH.txt  > Datacard_combined_M2000_SL_FH.txt
    # combineCards.py SL_M2600=Datacard_combined_M2600_SL.txt FH_M2600=Datacard_combined_M2600_FH.txt  > Datacard_combined_M2600_SL_FH.txt
    # combineCards.py SL_M3000=Datacard_combined_M3000_SL.txt FH_M3000=Datacard_combined_M3000_FH.txt  > Datacard_combined_M3000_SL_FH.txt

fi

if [ ${WhichSamples} -eq 4 ]
  then
    combine -M AsymptoticLimits -m 125 -n M300_SL_FH Datacard_combined_M300_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M300_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_SL_FH Datacard_combined_M500_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M500_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_SL_FH Datacard_combined_M700_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M700_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_SL_FH Datacard_combined_M1000_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M1000_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_SL_FH Datacard_combined_M1500_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M1500_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_SL_FH Datacard_combined_M2000_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M2000_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_SL_FH Datacard_combined_M2600_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M2600_SL_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_SL_FH Datacard_combined_M3000_SL_FH.txt --run expected --rMax 500000 >datacard_limits_M3000_SL_FH.log 2>&1 

    # combine -M AsymptoticLimits -m 125 -n M300_1jet Datacard_M300_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M300_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_1jet Datacard_M500_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M500_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_1jet Datacard_M700_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M700_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_1jet Datacard_M1000_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M1000_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_1jet Datacard_M1500_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M1500_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_1jet Datacard_M2000_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M2000_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_1jet Datacard_M2600_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M2600_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_1jet Datacard_M3000_1jet_FH.txt --run expected --rMax 50000 >datacard_limits_M3000_1jet.log 2>&1 
    
    # combine -M AsymptoticLimits -m 125 -n M300_2jets_3jets Datacard_M300_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M300_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_2jets_3jets Datacard_M500_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M500_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_2jets_3jets Datacard_M700_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M700_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_2jets_3jets Datacard_M1000_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M1000_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_2jets_3jets Datacard_M1500_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M1500_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_2jets_3jets Datacard_M2000_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M2000_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_2jets_3jets Datacard_M2600_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M2600_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_2jets_3jets Datacard_M3000_2jets_3jets_FH.txt --run expected --rMax 50000 >datacard_limits_M3000_2jets_3jets.log 2>&1   

    # combine -M AsymptoticLimits -m 125 -n M300_4jets Datacard_M300_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M300_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_4jets Datacard_M500_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M500_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_4jets Datacard_M700_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M700_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_4jets Datacard_M1000_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M1000_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_4jets Datacard_M1500_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M1500_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_4jets Datacard_M2000_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M2000_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_4jets Datacard_M2600_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M2600_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_4jets Datacard_M3000_4jets_FH.txt --run expected --rMax 50000 >datacard_limits_M3000_4jets.log 2>&1 

    # combine -M AsymptoticLimits -m 125 -n M300_FH Datacard_combined_M300_FH.txt --run expected --rMax 50000 >datacard_limits_M300_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_FH Datacard_combined_M500_FH.txt --run expected --rMax 50000 >datacard_limits_M500_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_FH Datacard_combined_M700_FH.txt --run expected --rMax 50000 >datacard_limits_M700_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_FH Datacard_combined_M1000_FH.txt --run expected --rMax 50000 >datacard_limits_M1000_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_FH Datacard_combined_M1500_FH.txt --run expected --rMax 50000 >datacard_limits_M1500_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_FH Datacard_combined_M2000_FH.txt --run expected --rMax 50000 >datacard_limits_M2000_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_FH Datacard_combined_M2600_FH.txt --run expected --rMax 50000 >datacard_limits_M2600_FH.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_FH Datacard_combined_M3000_FH.txt --run expected --rMax 50000 >datacard_limits_M3000_FH.log 2>&1 

    # combine -M AsymptoticLimits -m 125 -n M300_1jet Datacard_M300_1jet_SL.txt --run expected --rMax 50000 >datacard_limits_M300_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_2jets Datacard_M500_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M500_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_2jets Datacard_M700_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M700_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_2jets Datacard_M1000_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M1000_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_2jets Datacard_M1500_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M1500_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_2jets Datacard_M2000_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M2000_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_2jets Datacard_M2600_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M2600_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_2jets Datacard_M3000_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M3000_2jets.log 2>&1 

    # combine -M AsymptoticLimits -m 125 -n M300_2jets Datacard_M300_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M300_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_2jets Datacard_M500_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M500_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_2jets Datacard_M700_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M700_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_2jets Datacard_M1000_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M1000_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_2jets Datacard_M1500_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M1500_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_2jets Datacard_M2000_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M2000_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_2jets Datacard_M2600_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M2600_2jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_2jets Datacard_M3000_2jets_SL.txt --run expected --rMax 50000 >datacard_limits_M3000_2jets.log 2>&1 

    # combine -M AsymptoticLimits -m 125 -n M300_SL Datacard_M300_2jets_SL_SL.txt --run expected --rMax 50000

    # combine -M AsymptoticLimits -m 125 -n M300_SL Datacard_combined_M300_SL.txt --run expected --rMax 50000 >datacard_limits_M300_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_SL Datacard_combined_M500_SL.txt --run expected --rMax 50000 >datacard_limits_M500_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_SL Datacard_combined_M700_SL.txt --run expected --rMax 50000 >datacard_limits_M700_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_SL Datacard_combined_M1000_SL.txt --run expected --rMax 50000 >datacard_limits_M1000_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_SL Datacard_combined_M1500_SL.txt --run expected --rMax 50000 >datacard_limits_M1500_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_SL Datacard_combined_M2000_SL.txt --run expected --rMax 50000 >datacard_limits_M2000_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_SL Datacard_combined_M2600_SL.txt --run expected --rMax 50000 >datacard_limits_M2600_SL.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_SL Datacard_combined_M3000_SL.txt --run expected --rMax 50000 >datacard_limits_M3000_SL.log 2>&1 
  
fi

if [ ${WhichSamples} -eq 5 ]
  # move limit root file to eos to further analysis 
  then
    # mv higgsCombineM*_1jet.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/1jet_limit
    # mv higgsCombineM*_2jets_3jets.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/2jets_3jets_limit
    # mv higgsCombineM*_4jets.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/4jets_limit
    # mv higgsCombineM*_FH.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/combined_limit/
    # mv higgsCombineM*_SL.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_SL/combined_limit/
    mv higgsCombineM*_SL_FH.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_SL/
    
fi
if [ ${WhichSamples} -eq 6 ]
  # sed chg content
  then
    cat BR_qqqq_datacard.txt >>Datacard_M300_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M300_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M500_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M700_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M1000_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M1500_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M2000_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M2600_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M3000_1jet_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M300_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M500_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M700_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M1000_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M1500_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M2000_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M2600_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M3000_2jets_3jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M300_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M500_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M700_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M1000_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M1500_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M2000_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M2600_4jets_FH.txt
    cat BR_qqqq_datacard.txt >>Datacard_M3000_4jets_FH.txt
    
    
fi