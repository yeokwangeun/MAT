#!/bin/bash

############### Edit this path to your own ###############
lst="./MAT/TrackEval/personpath_sample/list.txt" # list of video sequences
gt="./MAT/TrackEval/data/gt/mot_challenge/PersonPath-test" # gt detection folder
samplelist="./MAT/TrackEval/personpath_sample" # output folder with indices to REMOVE


# get indices to remove
while read -r line; do
    gtfile="$gt/$line/gt/gt.txt"
    grep ",1000000000" $gtfile | awk -F ',' '{print $1}' > "$samplelist/$line.txt"
done < "$lst"