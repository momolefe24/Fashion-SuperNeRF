#!/bin/bash
args=("$@")
shift 4
root_dir="${args[0]}"
person="${args[1]}"
clothing="${args[2]}"
filename="${args[3]}"
echo "root_dir: ${root_dir}"
echo "person: ${person}"
echo "clothing: ${clothing}"
echo "filename: ${filename}"
#./openpose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
echo "${root_dir}/${filename}"
image_file="${root_dir}/${filename}"
openpose_input_image="../openpose/temp_input"
openpose_output_json="../openpose/temp_json"
openpose_output_image="../openpose/temp_output"
if [ ! -d "${openpose_input_image}" ]
then
  mkdir -p "${openpose_input_image}"
fi

if [ ! -d "${openpose_output_json}" ]
then
  mkdir -p "${openpose_output_json}"
fi

if [ ! -d "${openpose_output_image}" ]
then
  mkdir -p "${openpose_output_image}"
fi

if [ ! -f "${openpose_input_image}/${filename}" ]
then
  cp "${root_dir}/${filename}" ${openpose_input_image}
fi

cd ../openpose
./build/examples/openpose/openpose.bin --image_dir ${openpose_input_image} --write_images ${openpose_output_image}/ --write_json ${openpose_output_json}/ --hand --display 0 --disable_blending --net_resolution -1x208
cd ../Research
openpose_results="${root_dir}/${person}_${clothing}"
if [ ! -d "${openpose_results}/openpose_img" ]
then
  mkdir "${openpose_results}/openpose_img"
fi

if [ ! -d "${openpose_results}/openpose_json" ]
then
  mkdir "${openpose_results}/openpose_json"
fi

png_filename=${filename//.jpg/_rendered.png}
json_filename=${filename//.jpg/_keypoints.json}
pwd
echo "${openpose_output_image}/${png_filename}"
if [ -f "${openpose_output_image}/${png_filename}" ]
then
  cp "${openpose_output_image}/${png_filename}" "${openpose_results}/openpose_img"
fi

if [ -f "${openpose_output_json}/${json_filename}" ]
then
  cp "${openpose_output_json}/${json_filename}" "${openpose_results}/openpose_json"
fi