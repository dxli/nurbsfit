#!/bin/bash


# Directory containing the pc files (PLY files)
input_folder="/home/lizonly/Downloads/noise_3dv/hand/"
#input_folder="/home/lizeth/Downloads/real_world_sampled_10k/"

# Output folder where you want the results to be saved
output_folder="/mnt/Chest/Repositories/NURBS_fit/data/"

# Path to the executable
executable="/mnt/Chest/Repositories/GoCoPP2/source/build/bin/Release/GoCoPP"

# Iterate over each PLY file in the input folder
for mesh_file in "$input_folder"/*.ply; do
    # Extract the base name of the file (without extension) to use for output
    base_name=$(basename "$mesh_file" .ply)

     base_name=$(basename "$mesh_file" .ply)

    # Set the output folder path
    output_shape_folder="$output_folder/$base_name"
#    -epsilon 0.001 --normal_deviation 0.5 --sigma 20  3800 prim

    # Check if the output folder already exists
    if [ -d "$output_shape_folder" ]; then
        echo "Skipping $mesh_file as the output folder $output_shape_folder already exists."
        continue
    fi
    # big sphere --epsilon 0.05 --normal_deviation 0.75 --sigma 500
    # Run the command with the current mesh file
    "$executable" "$mesh_file" --sigma 20 --vg --alpha --hull --out "$output_folder"

done
