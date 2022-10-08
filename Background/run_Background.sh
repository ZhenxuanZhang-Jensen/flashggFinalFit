############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "options:"
   echo "h     Print this Help."
   echo "f     Print flag we need to run."
   echo
}

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Set flag
flag="nothing"
# Get the options
while getopts ":hf:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      f) # Enter a name
         flag=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done
########### begin your code ####################
if [ "$flag" = "BkgFit" ]; then
   echo "running $flag"
   python RunBackgroundScripts.py --inputConfig config_FH_1jet.py --mode fTestParallel
   python RunBackgroundScripts.py --inputConfig config_FH_2jets_3jets.py --mode fTestParallel
   python RunBackgroundScripts.py --inputConfig config_FH_4jets.py --mode fTestParallel
fi
if [ "$flag" = "cpBkg_ws" ]; then
    # mv the root file to different ws
   echo "running $flag"
   cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws/Data_FH_2017_cat_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet/allData.root
   cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws/Data_FH_2017_cat_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets/allData.root
   cp /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws/Data_FH_2017_cat_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets/allData.root
   
   
fi
if [ "$flag" = "cpBkg" ]; then
    # need to mv the output file(CMS-HGG_multipdf_*.root) to */ws/
   echo "running $flag"
   cp outdir_FH_1jet/CMS-HGG_multipdf_RECO_untagged_1jet.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_1jet/CMS-HGG_multipdf_RECO_untagged_1jet_2017.root
   cp outdir_FH_2jets_3jets/CMS-HGG_multipdf_RECO_untagged_2jets_3jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_2jets_3jets/CMS-HGG_multipdf_RECO_untagged_2jets_3jets_2017.root
   cp outdir_FH_4jets/CMS-HGG_multipdf_RECO_untagged_4jets.root /eos/user/z/zhenxuan/hhwwgg_root/hhwwgg_root_FH/ws_4jets/CMS-HGG_multipdf_RECO_untagged_4jets_2017.root
   
fi
