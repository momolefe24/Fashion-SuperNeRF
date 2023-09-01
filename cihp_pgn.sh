#!/bin/bash
args=("$@")
shift 4
source /home-mscluster/mmolefe/miniconda3/bin/activate
home_dir="/home-mscluster/mmolefe"
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
CIHP_PGN_dir="${home_dir}/Playground/CIHP_PGN"
inference_pipeline_dir="${home_dir}/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/inference_pipeline"
cihp_pgn_root_dir="${inference_pipeline_dir}/${root_dir}"
cd $CIHP_PGN_dir
echo "CIHP_PGN root dir: ${cihp_pgn_root_dir}"
python3 inf_pgn.py --root_dir "${cihp_pgn_root_dir}" --person ${person} --clothing ${clothing} --filename ${filename}
echo "Done."

#'./cihp_pgncihp_pgn.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg'
# Example:
#pwd
#cd CIHP_PGN
#pwd
#python3 inf_pgn.py --name "CIHP_PGN" --root_dir "data" --data_source "rail" --person "julian" --clothing "gray_long_sleeve" --image 27 --extension jpg
#python3 inf_pgn.py --name "CIHP_PGN" --root_dir "data" --data_source "rail" --person "julian" --clothing "gray_long_sleeve" --image 27
#./cihp_pgn.sh data rail julian gray_long_sleeve 27 jpg
