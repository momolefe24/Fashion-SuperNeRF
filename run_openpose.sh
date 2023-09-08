#!/bin/bash
#!/bin/bash
args=("$@")
mscluster_path="Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/inference_pipeline"
data_path="data/rail/temp"
person="${args[0]}"
clothing="${args[1]}"
filename="${args[2]}"
echo "person: ${person}"
echo "clothing: ${clothing}"
echo "filename: ${filename}"

cd /home/mo/Playground/Artificial\ Intelligence/Computer\ Vision/openpose/
./build/examples/openpose/openpose.bin --image_dir $person --hand --write_json "${person}_json" --write_images "${person}_output" --disable_blending --display 0
png_filename=${filename//.jpg/_rendered.png}
json_filename=${filename//.jpg/_keypoints.json}
output_png="${person}_output/${png_filename}"
output_json="${person}_json/${json_filename}"
openpose_img_save_dir="$mscluster_path/$data_path/${person}_${clothing}/openpose_img"
openpose_json_save_dir="$mscluster_path/$data_path/${person}_${clothing}/openpose_json"
rsync $output_png mscluster:~/"${mscluster_path}/$person_$clothing/"
echo "Openpose JSON: $openpose_json_save_dir"
echo "Output: ${output_png}"

if [ ! -d "${openpose_json_save_dir}" ]
then
  mkdir -p "${openpose_json_save_dir}"
fi

if [ ! -d "${openpose_img_save_dir}" ]
then
  mkdir -p "${openpose_img_save_dir}"
fi

if [ -f "${output_png}" ]
then
  rsync $output_png mscluster:~/"${openpose_img_save_dir}"
fi

if [ -f "${output_json}" ]
then
  rsync $output_json mscluster:~/"${openpose_json_save_dir}"
fi