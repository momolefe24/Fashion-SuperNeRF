#!/bin/bash
args=("$@")
shift 4
source /home/molefe/miniconda3/bin/activate
root_dir="${args[0]}"
person="${args[1]}"
clothing="${args[2]}"
filename="${args[3]}"
echo "root_dir: ${root_dir}"
echo "person: ${person}"
echo "clothing: ${clothing}"
echo "filename: ${filename}"
conda activate tf
conda env list
python3 CIHP_PGN/inf_pgn.py --root_dir ${root_dir} --person ${person} --clothing ${clothing} --filename ${filename}


#'./cihp_pgncihp_pgn.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg'
# Example:
#pwd
#cd CIHP_PGN
#pwd
#python3 inf_pgn.py --name "CIHP_PGN" --root_dir "data" --data_source "rail" --person "julian" --clothing "gray_long_sleeve" --image 27 --extension jpg
#python3 inf_pgn.py --name "CIHP_PGN" --root_dir "data" --data_source "rail" --person "julian" --clothing "gray_long_sleeve" --image 27
#./cihp_pgn.sh data rail julian gray_long_sleeve 27 jpg
