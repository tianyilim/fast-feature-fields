#!/bin/bash
set -e

DOWNLOAD_PATH=$1
mkdir -p $DOWNLOAD_PATH
mkdir -p $DOWNLOAD_PATH/raw

DOWNLOAD_URLS=(
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_3_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_5_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_6_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_7_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_8_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_9_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_10_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_11_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_12_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_1_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_2_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_3_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_4_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_9_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_11_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_12_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_13_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_14_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_45_16_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_1_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_2_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_3_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_5_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_6_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_9_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_10_davis.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_45_1_davis_with_gt.zip
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_45_2_davis.zip
)

for url in "${DOWNLOAD_URLS[@]}"
do
    seq_name=$(basename "$url" .zip)
    echo "Downloading $seq_name"

    mkdir -p $DOWNLOAD_PATH/$seq_name
    if [ -e $DOWNLOAD_PATH/raw/$seq_name.zip ];  then
        echo "$seq_name already downloaded."
    else
        wget $url -P $DOWNLOAD_PATH/raw
    fi

    unzip -o $DOWNLOAD_PATH/raw/$seq_name.zip events.txt -d $DOWNLOAD_PATH/$seq_name
done
