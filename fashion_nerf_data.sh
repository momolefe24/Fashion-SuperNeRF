#!/bin/bash

# Check if at least one folder name is provided
if [ $# -eq 0 ]; then
    echo "Error: No folder names provided."
    exit 1
fi

# Array of folder names
folders=("$@")

# Function to run Python script and move the output file
run_python_and_move() {
    folder=$1
    output_file="${folder}.log"
    echo "Running Python script for folder '$folder'. Output will be saved in '$output_file'."

    echo "${folder} has transforms.json"
    # Create a new temporary directory for each process
    temp_dir=$(mktemp -d "tempdir_${folder}_XXXXX")

#    # Change the working directory to the folder before running the Python script
#    pushd "$folder" > /dev/null

    # Run the Python script in the background and redirect output to the file
#    python test.py --person "$folder" > "$output_file" 2>&1
    source /home/mo/miniconda3/bin/activate
    conda activate NeRF

    python3 colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images "${folder}" --out "transforms_${folder}.json"> "$output_file" 2>&1
    python3 colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images "${folder}"

    # Check if the Python script generated transforms.json file
    if [ -f "transforms_${folder}.json" ]; then
        # Move the transforms.json file to the temporary directory
        mv "transforms_${folder}.json" "$temp_dir"
    fi

}

# Loop through each folder and run the Python script with background process
for folder in "${folders[@]}"; do
    # Check if the folder exists
    if [ ! -d "$folder" ]; then
        echo "Error: Folder '$folder' not found."
    else
        run_python_and_move "$folder" &
    fi
done

echo "All Python processes are running in the background."