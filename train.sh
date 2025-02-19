#!/bin/bash

base_path="data"
output_dir="output3"

for folder_path in "$base_path"/*/; do
    folder_name=$(basename "$folder_path")

    if [[ "$folder_name" == *_8 || "$folder_name" == *_2 ]]; then
        # echo "Process train: $folder_name"
        # python train.py -s "$folder_path" -m "$output_dir/$folder_name"

        test_folder_name="${folder_name%_*}" 

        echo "Process render: $folder_name"
        python render.py -s "$base_path/$test_folder_name" -m "$output_dir/$folder_name"

        echo "Process metrics: $folder_name"
        python metrics.py -m "$output_dir/$folder_name"
    fi
done

echo "ALL DONE!"

