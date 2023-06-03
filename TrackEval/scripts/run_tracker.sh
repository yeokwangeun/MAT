#!/bin/bash
# Script to run tracker and convert to MOT format for evaluation. Evaluation is not done here!

# !! Make sure you replaced all 3 < path > in the for loop that fits your system !! 

# Usage: bash run_tracker.sh <video source folder> <output folder> <model name> <mode> 
# Example: bash run_tracker.sh /home/username/AnimalTrack/train/videos /home/username/AnimalTrack/train/result deepsort train 

# Check if all required arguments are provided
if [ $# -ne 4 ]; then
  echo "Usage: bash run_tracker.sh <video source folder> <output folder> <model name> <mode>"
  exit 1
fi

src=$1 # Folder with videos (ex. duck_5.mp4) to track
out=$2 # Folder to save raw tracker outputs  
model=$3 # model name (ex. deepsort)
mode=$4 # train or test 
log="$out/log_$model.txt" 

###############################################
# EDIT TO YOUR OWN MAT PATH
mat="" # ex) ./MAT
trackeval="$mat/TrackEval" # ex) ./MAT/TrackEval
###############################################

echo "Video source folder: $src" >> $log
echo "Output folder: $out" >> $log
echo "Model: $model" >> $log
echo "Mode: $mode" >> $log

# Create output folder for evaluation
mkdir -p "$trackeval/data/trackers/mot_challenge/AnimalTrack-$mode/$model/data"

for vid in $src/*.mp4; do
    echo "Analyzing $vid, starting at $(date)" >> $log
    vid_name=$(basename $vid .mp4) # ex. duck_5
    mkdir -p $out/$vid_name
    mkdir -p $trackeval/data/trackers/mot_challenge/AnimalTrack-$mode/$model/data --verbose
    
    # Run tracker - outputs will be saved under $out/$vid_name/raw_tracker.csv
    ###############################################
    # EDIT cfg, weights PATH TO YOUR OWN  
    python3 $mat/track.py --source $vid --output $out/$vid_name --save-txt \
    --cfg < path to yolov3.cfg > --weights < path to yolov3.weights >
    ###############################################
    echo "Done tracking $vid, ending at $(date)" >> $log

    # Convert to MOT format - outputs will be saved in MAT/TrackEval/data/trackers/...
    python3 $trackeval/utils/convert_tracker_format.py \
    --raw_tracker_path $out/$vid_name/raw_tracker.csv \
    --convert_path $trackeval/data/trackers/mot_challenge/AnimalTrack-$mode/$model/data/$vid_name.txt
    echo "Done converting $vid_name, ending at $(date)" >> $log
done

echo "Done tracking all videos, ending at $(date)" >> $log