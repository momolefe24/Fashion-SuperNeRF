#!/bin/bash
args=("$@")
shift 4
root_dir="${args[0]}"
person="${args[1]}"
clothing="${args[2]}"
filename="${args[3]}"
folder_name="${person}_${clothing}"
echo "root_dir: ${root_dir}"
echo "person: ${person}"
echo "clothing: ${clothing}"
echo "filename: ${filename}"
#./openpose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
echo "${root_dir}/${folder_name}/${filename}"
image_file="${root_dir}/${filename}"
image_path="${root_dir}/${folder_name}/${filename}"
echo "image path: $image_path"
rail_dir=/Playground/Artificial\ Intelligence/Computer\ Vision/openpose/${person}/
echo "rail dir: $rail_dir"
rsync -hacvzP $image_path myrail:~/"$rail_dir"
ssh myrail 'bash -s' < run_openpose.sh $person $clothing $filename
