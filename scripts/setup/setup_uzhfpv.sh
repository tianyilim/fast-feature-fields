#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "1 argument expected: path to UZH-FPV dataset."
    echo "$# arguments provided."
    echo "Call this script from the root directory; i.e. bash scripts/setup/setup_uzh_fpv.sh [DATA_DIR]."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1 # Where we store the uzh-fpv dataset

# Download ZIP files, and extract the event files (as txt)
bash scripts/download/download_uzhFpv.sh $1

# Process these txt files into h5 files.
python3 scripts/download/uzh_txt_to_h5.py $1

# Iterate through all h5 files and generate the timestamps npy files.
for folder in $base_path/*; do
    seq_name=$(basename $folder)
    seq_h5="$folder"/"$seq_name".h5
    if [ -e "$seq_h5" ]; then
        # echo "EXISTS: $seq_h5"
        python3 scripts/generate_ts.py --data_h5 $seq_h5 --dataset "uzhfpv"
    else
        echo "MISSING: $seq_h5"
    fi
    # Make symlink to our data folder
    ln -s $folder data/$seq_name
done
