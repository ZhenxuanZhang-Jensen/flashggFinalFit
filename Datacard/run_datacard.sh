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
    python RunYields.py --inputWSDirMap 2017=/eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_1jet_l200 --sigModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_gghh_M3000_1jet_l200 --bkgModelWSDir /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH_custom/ws_1jet_l200 --cats auto --procs auto --batch local --ext M3000_1jet_l200 >M3000_FH_1jet_yields.log 2>&1

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
    python makeDatacard.py --years 2017 --prune --ext 'M1100_FH_2017_FHSLCombine_FH_boosted_low_purity'  'M1100_SL_2017_FHSLCombine_FH_boosted_low_purity' --output Datacard_test
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
    
    python makeDatacard.py --years 2017 --prune --ext 'M3000_1jet_l200' --output Datacard_M3000_1jet_l200_FH
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
    # combineCards.py M1100_2017_FHSLCombine_FH_boosted_low_purity=Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_low_purity.txt M1100_2017_FHSLCombine_FH_boosted_high_purity=Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_high_purity.txt > Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_combined.txt
    combineCards.py M1100_2017_FHSLCombine_FH_boosted_low_purity=Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_low_purity.txt M1100_2017_FHSLCombine_FH_boosted_high_purity=Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_high_purity.txt M1100_2017_FHSLCombine_SL_merged_boosted_cat=Datacard_M1100_FHSL_2017_FHSLCombine_SL_merged_boosted_cat.txt > Datacard_M1100_FH_2017_FHSLCombine_allCategory_combined.txt

    # combineCards.py M1100_2017_FHSLCombine_FH_boosted_low_purity=Datacard_M1100_SL_2017_FHSLCombine_FH_boosted_low_purity.txt M1100_2017_FHSLCombine_FH_boosted_high_purity=Datacard_M1100_SL_2017_FHSLCombine_FH_boosted_high_purity.txt > Datacard_M1100_SL_2017_FHSLCombine_FH_boosted_combined.txt

    combineCards.py Datacard_M1100_FH_2017_FHSLCombine_FH_boosted_low_purity.txt Datacard_M1100_FH_2017_FHSLCombine_FH_boosted_low_purity.txt > Datacard_test.txt

    combineCards.py M1100_2017__FH_fullyresolved=Datacard_M1700_2017_FHSamples_SL_fullyresovled_cat.txt M1100_2017__SL_fullyresolved=Datacard_M1700_2017_FHSamples_FH_fully_resovled_cat.txt > Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_combined_test.txt
  
    combine -M AsymptoticLimits -m 125 -n M1100_FHSL Datacard_M1100_FH_2017_FHSLCombine_FH_boosted_combined.root --setParameters mask_signal=1 --run expected >datacard_limits_FHSL_1jets_M1400.log 2>&1 

    combineCards.py

    combineCards.py FHSL_1jets_M2000_cat0=Datacard_M2000_1jets_cat0_FHSL.txt FHSL_1jets_M2000_cat1=Datacard_M2000_1jets_cat1_FHSL.txt FHSL_1jets_M2000_cat2=Datacard_M2000_1jets_cat2_FHSL.txt FHSL_1jets_M2000_cat3=Datacard_M2000_1jets_cat3_FHSL.txt FHSL_1jets_M2000_cat4=Datacard_M2000_1jets_cat4_FHSL.txt FHSL_1jets_M2000_cat5=Datacard_M2000_1jets_cat5_FHSL.txt FHSL_1jets_M2000_cat6=Datacard_M2000_1jets_cat6_FHSL.txt FHSL_1jets_M2000_cat7=Datacard_M2000_1jets_cat7_FHSL.txt FHSL_1jets_M2000_cat8=Datacard_M2000_1jets_cat8_FHSL.txt > Datacard_combined_FHSL_1jets_M2000.txt
    combine -M AsymptoticLimits -m 125 -n FH_M1100_FH_boosted_combined Datacard_M1100_FHSL_2017_FHSLCombine_FH_boosted_high_purity.txt --run expected >datacard_limits_FHSL_1jets_M2000.log 2>&1 


    combineCards.py FHSL_1jets_M2400_cat0=Datacard_M2400_1jets_cat0_FHSL.txt FHSL_1jets_M2400_cat1=Datacard_M2400_1jets_cat1_FHSL.txt FHSL_1jets_M2400_cat2=Datacard_M2400_1jets_cat2_FHSL.txt FHSL_1jets_M2400_cat3=Datacard_M2400_1jets_cat3_FHSL.txt FHSL_1jets_M2400_cat4=Datacard_M2400_1jets_cat4_FHSL.txt FHSL_1jets_M2400_cat5=Datacard_M2400_1jets_cat5_FHSL.txt FHSL_1jets_M2400_cat6=Datacard_M2400_1jets_cat6_FHSL.txt > Datacard_combined_FHSL_1jets_M2400.txt
    combine -M AsymptoticLimits -m 125 -n FHSL_1jets_M2400 Datacard_combined_FHSL_1jets_M2400.txt --run expected >datacard_limits_FHSL_1jets_M2400.log 2>&1 

    combineCards.py FHSL_1jets_M1000_cat0=Datacard_M1000_1jets_cat0_FHSL.txt FHSL_1jets_M1000_cat1=Datacard_M1000_1jets_cat1_FHSL.txt FHSL_1jets_M1000_cat2=Datacard_M1000_1jets_cat2_FHSL.txt FHSL_1jets_M1000_cat3=Datacard_M1000_1jets_cat3_FHSL.txt FHSL_1jets_M1000_cat4=Datacard_M1000_1jets_cat4_FHSL.txt FHSL_1jets_M1000_cat5=Datacard_M1000_1jets_cat5_FHSL.txt FHSL_1jets_M1000_cat6=Datacard_M1000_1jets_cat6_FHSL.txt FHSL_1jets_M1000_cat7=Datacard_M1000_1jets_cat7_FHSL.txt FHSL_1jets_M1000_cat8=Datacard_M1000_1jets_cat8_FHSL.txt FHSL_1jets_M1000_cat9=Datacard_M1000_1jets_cat9_FHSL.txt > Datacard_combined_FHSL_1jets_M1000.txt
    combine -M AsymptoticLimits -m 125 -n FHSL_1jets_M1000 Datacard_combined_FHSL_1jets_M1000.txt --run expected >datacard_limits_FHSL_1jets_M1000.log 2>&1 

    combineCards.py FHSL_1jets_M3000_cat0=Datacard_M3000_1jets_cat0_FHSL.txt FHSL_1jets_M3000_cat1=Datacard_M3000_1jets_cat1_FHSL.txt FHSL_1jets_M3000_cat2=Datacard_M3000_1jets_cat2_FHSL.txt FHSL_1jets_M3000_cat3=Datacard_M3000_1jets_cat3_FHSL.txt FHSL_1jets_M3000_cat4=Datacard_M3000_1jets_cat4_FHSL.txt FHSL_1jets_M3000_cat5=Datacard_M3000_1jets_cat5_FHSL.txt FHSL_1jets_M3000_cat6=Datacard_M3000_1jets_cat6_FHSL.txt > Datacard_combined_FHSL_1jets_M3000.txt
    combine -M AsymptoticLimits -m 125 -n FHSL_1jets_M3000 Datacard_combined_FHSL_1jets_M3000.txt --run expected >datacard_limits_FHSL_1jets_M3000.log 2>&1 

    combineCards.py FHSL_1jets_M1500_cat0=Datacard_M1500_1jets_cat0_FHSL.txt FHSL_1jets_M1500_cat1=Datacard_M1500_1jets_cat1_FHSL.txt FHSL_1jets_M1500_cat2=Datacard_M1500_1jets_cat2_FHSL.txt FHSL_1jets_M1500_cat3=Datacard_M1500_1jets_cat3_FHSL.txt FHSL_1jets_M1500_cat4=Datacard_M1500_1jets_cat4_FHSL.txt FHSL_1jets_M1500_cat5=Datacard_M1500_1jets_cat5_FHSL.txt FHSL_1jets_M1500_cat6=Datacard_M1500_1jets_cat6_FHSL.txt FHSL_1jets_M1500_cat7=Datacard_M1500_1jets_cat7_FHSL.txt FHSL_1jets_M1500_cat8=Datacard_M1500_1jets_cat8_FHSL.txt > Datacard_combined_FHSL_1jets_M1500.txt
    combine -M AsymptoticLimits -m 125 -n FHSL_1jets_M1500 Datacard_combined_FHSL_1jets_M1500.txt --run expected >datacard_limits_FHSL_1jets_M1500.log 2>&1 

    combineCards.py FHSL_1jets_M1100_cat0=Datacard_M1100_1jets_cat0_FHSL.txt FHSL_1jets_M1100_cat1=Datacard_M1100_1jets_cat1_FHSL.txt FHSL_1jets_M1100_cat2=Datacard_M1100_1jets_cat2_FHSL.txt FHSL_1jets_M1100_cat3=Datacard_M1100_1jets_cat3_FHSL.txt FHSL_1jets_M1100_cat4=Datacard_M1100_1jets_cat4_FHSL.txt FHSL_1jets_M1100_cat5=Datacard_M1100_1jets_cat5_FHSL.txt FHSL_1jets_M1100_cat6=Datacard_M1100_1jets_cat6_FHSL.txt FHSL_1jets_M1100_cat7=Datacard_M1100_1jets_cat7_FHSL.txt FHSL_1jets_M1100_cat8=Datacard_M1100_1jets_cat8_FHSL.txt FHSL_1jets_M1100_cat9=Datacard_M1100_1jets_cat9_FHSL.txt > Datacard_combined_FHSL_1jets_M1100.txt
    combine -M AsymptoticLimits -m 125 -n 2017_FHSLCombine_allCategory_combined Datacard_M1100_FH_2017_FHSLCombine_allCategory_combined.txt --run expected >datacard_limits_FHSL_1jets_M1100.log 2>&1 


    # combineCards.py FHSL_1jets_M3000_cat0=Datacard_M3000_1jets_cat0_FHSL.txt FHSL_1jets_M3000_cat1=Datacard_M3000_1jets_cat1_FHSL.txt FHSL_1jets_M3000_cat2=Datacard_M3000_1jets_cat2_FHSL.txt FHSL_1jets_M3000_cat3=Datacard_M3000_1jets_cat3_FHSL.txt   > Datacard_combined_FHSL_1jets_M3000.txt
    # combineCards.py SL_M500=Datacard_combined_M500_SL.txt FH_M500=Datacard_combined_M500_FH.txt  > Datacard_combined_M500_SL_FH.txt
    # combine SL
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
    combine -M AsymptoticLimits -m 125 -n Datacard_M1100_2017_FHSLSamples_FH_boosted.txt --run expected --rMax 500000 >datacard_limits_M300_SL_FH.log 2>&1 
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
    combine -M AsymptoticLimits -m 125 -n M3000_FH Datacard_M3000_1jet_l200_FH.txt --run expected --rMax 50000 >datacard_limits_M3000_FH.log 2>&1 

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