WhichSamples=${1}
if [ ${WhichSamples} -eq 0 ]
  then
    echo 'run plot_limits'
    python plot_limits.py -a --unit 'fb' --ymin 500 --ymax 10000000 --year 2017 --yboost 0.1 --resultType "HH" -l "combined_log"
fi


# python Plot_resonant.py -a  --resultType HH --unit fb --ymin 1 --ymax 1000 --yboost 0.1 --year Run2 
# python Plot_resonant.py -a  --resultType BBgg --unit fb --ymin 0.01 --ymax 10 --yboost 0.1 --year Run2