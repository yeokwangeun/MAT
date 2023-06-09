#!/bin/bash

# 1. load bef. sampling tracker output (aft. conversion to MOT-format)
# 2. based on gt sampling data file ($gt/*.txt), remove indices from $bef, save as $aft

################# EDIT THESE PATHS #################
gt="./MAT/TrackEval/personpath_sample" # gt sampling data folder 
bef="./MAT/TrackEval/data/trackers/mot_challenge/PersonPath-test/model1/data" # tracker output folder, before sampling
aft="./MAT/TrackEval/data/trackers/mot_challenge/PersonPath-test/model1_sample/data" # tracker output folder, after sampling
####################################################

mkdir -p $aft

for file in $bef/*.txt; do
    filename=$(basename "$file" .txt)
    echo $filename
    
    # Usage: python3 sample_frames.py --gt_sample <gt_sample_path> --bef_sample <bef_sample_path> --aft_sample <aft_sample_path>    
    python3 ./MAT/TrackEval/utils/sample_frames.py --gt_sample "$gt/$filename.txt" --bef_sample "$bef/$filename.txt" --aft_sample "$aft/$filename.txt"
done
