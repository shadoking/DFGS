#!/bin/bash

base_path="data"


for folder_path in "$base_path"/*/; do
    folder_name=$(basename "$folder_path")

    if [[ "$folder_name" == *_8 || "$folder_name" == *_2 ]]; then
    # if [[ "$folder_name" != *_8 && "$folder_name" != *_2 ]]; then
        echo "Process train: $folder_name"
        python script_for_dust3r.py -s "$folder_path"
    fi
done

echo "ALL DONE!"

