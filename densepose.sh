#!/bin/bash
args=("$@")
shift 4
source /home-mscluster/mmolefe/miniconda3/bin/activate
home_dir="/home-mscluster/mmolefe"
detectron_root="Playground/detectron2"
root_dir=$(pwd)
output_dir="densepose"
data_dir="${args[0]}"
person="${args[1]}"
clothing="${args[2]}"
filename="${args[3]}"
densepose_root_dir="${home_dir}/${detectron_root}/projects/DensePose"
echo "root_dir: ${densepose_root_dir}"
echo "data_dir: ${data_dir}"
echo "person: ${person}"
echo "clothing: ${clothing}"
echo "filename: ${filename}"
conda activate detectron
conda env list
inference_pipeline_dir="${home_dir}/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/inference_pipeline"
densepose_results="${data_dir}/${person}_${clothing}"
#mkdir -p "output/${output_dir}"
cd ${densepose_root_dir}
inference_data_dir="${inference_pipeline_dir}/${data_dir}/${person}_${clothing}"
densepose_input="${inference_data_dir}/image/${filename}"
densepose_output="${inference_data_dir}/image-densepose/${filename}"
echo "densepose_input: ${densepose_input}"
echo "densepose_output: ${densepose_output}"
python3 apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml densepose_rcnn_R_50_FPN_s1x.pkl "${densepose_input}" dp_segm -v --output "${densepose_output}"
apply_net_filename=${densepose_output//./.0001.}
if [ -f "$apply_net_filename" ]
 then
   echo "Filename in densepose ${apply_net_filename} "
   mv "$apply_net_filename" "$densepose_output"
   echo "Filename in densepose ${apply_net_filename} "
fi

# echo "Densepose filename: ${densepose_filename}"

# python3 apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml densepose_rcnn_R_50_FPN_s1x.pkl "${densepose_input}" dp_segm -v --output "${apply_net_output}"
# artifact_densepose_input=${apply_net_output//.jpg/.0001.jpg}
# echo "artifact_densepose_input: $artifact_densepose_input"
# python3 remove_artifacts.py --save_dir "output/${output_dir}" --filename ${artifact_densepose_input}
# if [ ! -d "${densepose_results}/densepose" ]
# then
#   echo "Making directory: ${densepose_results}/densepose"
#   mkdir "${densepose_results}/densepose"
# fi

# if [ -f "${artifact_densepose_input}" ]
# then
#     echo "Moving $artifact_densepose_input to ${densepose_results}/densepose/${filename} "
#     mv "$artifact_densepose_input" "${densepose_results}/densepose/${filename}"
# fi

# ./densepose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg

#pwd
#cd CIHP_PGN
#pwd
#python3 inf_pgn.py --name "CIHP_PGN" --root_dir "data" --data_source "rail" --person "julian" --clothing "gray_long_sleeve" --image 27 --extension jpg
#python3 inf_pgn.py --name "CIHP_PGN" --root_dir "data" --data_source "rail" --person "julian" --clothing "gray_long_sleeve" --image 27
#./cihp_pgn.sh data rail julian gray_long_sleeve 27 jpg
