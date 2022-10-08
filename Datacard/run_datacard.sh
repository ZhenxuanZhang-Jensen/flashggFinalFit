WhichSamples=${1}


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
    combineCards.py Datacard_M300_1jet_FH.txt Datacard_M300_2jets_3jets_FH.txt Datacard_M300_4jets_FH.txt  > Datacard_combined_M300_FH.txt
    combineCards.py Datacard_M500_1jet_FH.txt Datacard_M500_2jets_3jets_FH.txt Datacard_M500_4jets_FH.txt  > Datacard_combined_M500_FH.txt
    combineCards.py Datacard_M700_1jet_FH.txt Datacard_M700_2jets_3jets_FH.txt Datacard_M700_4jets_FH.txt  > Datacard_combined_M700_FH.txt
    combineCards.py Datacard_M1000_1jet_FH.txt Datacard_M1000_2jets_3jets_FH.txt Datacard_M1000_4jets_FH.txt  > Datacard_combined_M1000_FH.txt
    combineCards.py Datacard_M1500_1jet_FH.txt Datacard_M1500_2jets_3jets_FH.txt Datacard_M1500_4jets_FH.txt  > Datacard_combined_M1500_FH.txt
    combineCards.py Datacard_M2000_1jet_FH.txt Datacard_M2000_2jets_3jets_FH.txt Datacard_M2000_4jets_FH.txt  > Datacard_combined_M2000_FH.txt
    combineCards.py Datacard_M2600_1jet_FH.txt Datacard_M2600_2jets_3jets_FH.txt Datacard_M2600_4jets_FH.txt  > Datacard_combined_M2600_FH.txt
    combineCards.py Datacard_M3000_1jet_FH.txt Datacard_M3000_2jets_3jets_FH.txt Datacard_M3000_4jets_FH.txt  > Datacard_combined_M3000_FH.txt

fi
if [ ${WhichSamples} -eq 4 ]
  then
    # combine -M AsymptoticLimits -m 125 -n M300_1jet Datacard_M300_1jet_FH.txt --run expected >datacard_limits_M300_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_1jet Datacard_M500_1jet_FH.txt --run expected >datacard_limits_M500_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_1jet Datacard_M700_1jet_FH.txt --run expected >datacard_limits_M700_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_1jet Datacard_M1000_1jet_FH.txt --run expected >datacard_limits_M1000_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_1jet Datacard_M1500_1jet_FH.txt --run expected >datacard_limits_M1500_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_1jet Datacard_M2000_1jet_FH.txt --run expected >datacard_limits_M2000_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_1jet Datacard_M2600_1jet_FH.txt --run expected >datacard_limits_M2600_1jet.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_1jet Datacard_M3000_1jet_FH.txt --run expected >datacard_limits_M3000_1jet.log 2>&1 
    
    # combine -M AsymptoticLimits -m 125 -n M300_2jets_3jets Datacard_M300_2jets_3jets_FH.txt --run expected >datacard_limits_M300_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_2jets_3jets Datacard_M500_2jets_3jets_FH.txt --run expected >datacard_limits_M500_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_2jets_3jets Datacard_M700_2jets_3jets_FH.txt --run expected >datacard_limits_M700_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_2jets_3jets Datacard_M1000_2jets_3jets_FH.txt --run expected >datacard_limits_M1000_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_2jets_3jets Datacard_M1500_2jets_3jets_FH.txt --run expected >datacard_limits_M1500_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_2jets_3jets Datacard_M2000_2jets_3jets_FH.txt --run expected >datacard_limits_M2000_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_2jets_3jets Datacard_M2600_2jets_3jets_FH.txt --run expected >datacard_limits_M2600_2jets_3jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_2jets_3jets Datacard_M3000_2jets_3jets_FH.txt --run expected >datacard_limits_M3000_2jets_3jets.log 2>&1   

    # combine -M AsymptoticLimits -m 125 -n M300_4jets Datacard_M300_4jets_FH.txt --run expected >datacard_limits_M300_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M500_4jets Datacard_M500_4jets_FH.txt --run expected >datacard_limits_M500_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M700_4jets Datacard_M700_4jets_FH.txt --run expected >datacard_limits_M700_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1000_4jets Datacard_M1000_4jets_FH.txt --run expected >datacard_limits_M1000_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M1500_4jets Datacard_M1500_4jets_FH.txt --run expected >datacard_limits_M1500_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2000_4jets Datacard_M2000_4jets_FH.txt --run expected >datacard_limits_M2000_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M2600_4jets Datacard_M2600_4jets_FH.txt --run expected >datacard_limits_M2600_4jets.log 2>&1 
    # combine -M AsymptoticLimits -m 125 -n M3000_4jets Datacard_M3000_4jets_FH.txt --run expected >datacard_limits_M3000_4jets.log 2>&1 

    combine -M AsymptoticLimits -m 125 -n M300_combined Datacard_combined_M300_FH.txt --run expected >datacard_limits_M300_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M500_combined Datacard_combined_M500_FH.txt --run expected >datacard_limits_M500_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M700_combined Datacard_combined_M700_FH.txt --run expected >datacard_limits_M700_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M1000_combined Datacard_combined_M1000_FH.txt --run expected >datacard_limits_M1000_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M1500_combined Datacard_combined_M1500_FH.txt --run expected >datacard_limits_M1500_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M2000_combined Datacard_combined_M2000_FH.txt --run expected >datacard_limits_M2000_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M2600_combined Datacard_combined_M2600_FH.txt --run expected >datacard_limits_M2600_combined.log 2>&1 
    combine -M AsymptoticLimits -m 125 -n M3000_combined Datacard_combined_M3000_FH.txt --run expected >datacard_limits_M3000_combined.log 2>&1 
  

    
fi
if [ ${WhichSamples} -eq 5 ]
  # move limit root file to eos to further analysis 
  then
    # mv higgsCombineM*_1jet.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/1jet_limit
    # mv higgsCombineM*_2jets_3jets.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/2jets_3jets_limit
    # mv higgsCombineM*_4jets.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/4jets_limit
    mv higgsCombineM*_combined*.AsymptoticLimits.mH125.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/combined_limit
    
    
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