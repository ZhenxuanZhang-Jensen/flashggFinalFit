#!/bin/bash

cd /afs/cern.ch/user/z/zhenxuan/CMSSW_10_6_20/src/flashggFinalFit/Combine

eval `scramv1 runtime -sh`

text2workspace.py Datacard_M3000_FHSL_2017_FHSLCombine_FH_fully_resovled_cat.txt -o Datacard_M3000_FHSL_2017_FHSLCombine_FH_fully_resovled_cat_mu_inclusive.root -m 125 higgsMassRange=122,128 