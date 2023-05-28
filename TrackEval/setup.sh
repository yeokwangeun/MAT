#!/bin/bash

# Setup the required file structure for TrackEval - I've already done this, NO NEED TO RUN IT AGAIN
# Get the detection results from the AnimalTrack dataset (just the gt.txt and seqinfo.ini files, not the images)
# Data will be stored in TrackEval/data

# !!! Before running this script, activate an environment with numpy installed !!!

############# EDIT HERE ############# 
animaltrack_dir="./MAT/AnimalTrack" # path to AnimalTrack dataset
trackeval_dir="./Programs/TrackEval" # path to TrackEval repo
#####################################
convert_py="$trackeval_dir/utils/convert_gt_format.py" # convert gt format script

# Directory setup
seqmap_dir=$trackeval_dir/data/gt/mot_challenge/seqmaps # metadata
mkdir -p $trackeval_dir/data/gt/mot_challenge/AnimalTrack-test # gt-test
mkdir -p $trackeval_dir/data/gt/mot_challenge/AnimalTrack-train # gt-train
mkdir -p $seqmap_dir 
mkdir -p $trackeval_dir/data/trackers/mot_challenge/AnimalTrack-test/DeepSORT/data # tracker-test
mkdir -p $trackeval_dir/data/trackers/mot_challenge/AnimalTrack-train/DeepSORT/data # tracker-train
mkdir -p $trackeval_dir/result # evaluation results
echo "Directory setup done in $trackeval_dir/data" 


# Get metadata for AnimalTrack
echo "name" > $seqmap_dir/AnimalTrack-test.txt
echo "name" > $seqmap_dir/AnimalTrack-train.txt
ls $animaltrack_dir/test >> $seqmap_dir/AnimalTrack-test.txt
ls $animaltrack_dir/train >> $seqmap_dir/AnimalTrack-train.txt

cat $seqmap_dir/AnimalTrack-test.txt | tail -n +2 > $seqmap_dir/AnimalTrack-all.txt
cat $seqmap_dir/AnimalTrack-train.txt | tail -n +2 >> $seqmap_dir/AnimalTrack-all.txt
sort $seqmap_dir/AnimalTrack-all.txt > $seqmap_dir/AnimalTrack-all.txt.tmp
echo "name" | cat - $seqmap_dir/AnimalTrack-all.txt.tmp > $seqmap_dir/AnimalTrack-all.txt; rm $seqmap_dir/AnimalTrack-all.txt.tmp
echo "Writing metadata done in $seqmap_dir"


# Get gt for AnimalTrack
while read -r line; do # test set
    seq_name=$line # ex. chicken_1
    gt_dir=$animaltrack_dir/test/$seq_name
    out_dir=$trackeval_dir/data/gt/mot_challenge/AnimalTrack-test/$seq_name

    mkdir -p $out_dir/gt
    cp $gt_dir/seqinfo.ini $out_dir/
    python $convert_py --raw_gt_path $gt_dir/gt/gt.txt --convert_path $out_dir/gt/gt.txt # convert file format to MOT17
done < <(cat $seqmap_dir/AnimalTrack-test.txt | tail -n +2)
echo "Got gt labels for test set in $trackeval_dir/data/gt/mot_challenge/AnimalTrack-test"

while read -r line; do # train set
    seq_name=$line 
    gt_dir=$animaltrack_dir/train/$seq_name
    out_dir=$trackeval_dir/data/gt/mot_challenge/AnimalTrack-train/$seq_name

    mkdir -p $out_dir/gt
    cp $gt_dir/seqinfo.ini $out_dir/
    python $convert_py --raw_gt_path $gt_dir/gt/gt.txt --convert_path $out_dir/gt/gt.txt # convert file format to MOT17
done < <(cat $seqmap_dir/AnimalTrack-train.txt | tail -n +2)
echo "Got gt labels for train set in $trackeval_dir/data/gt/mot_challenge/AnimalTrack-train"



# # Prepare mock tracker detection results for AnimalTrack, just to test the evaluation script
# while read -r line; do # test set
#     seq_name=$line # ex. chicken_1
#     in_dir=$trackeval_dir/data/gt/mot_challenge/AnimalTrack-test/$seq_name/gt/gt.txt
#     out_dir=$trackeval_dir/data/trackers/mot_challenge/AnimalTrack-test/DeepSORT/data/$seq_name.txt

#     cp $in_dir $out_dir
# done < <(cat $seqmap_dir/AnimalTrack-test.txt | tail -n +2)

# while read -r line; do # test set
#     seq_name=$line # ex. chicken_1
#     in_dir=$trackeval_dir/data/gt/mot_challenge/AnimalTrack-train/$seq_name/gt/gt.txt
#     out_dir=$trackeval_dir/data/trackers/mot_challenge/AnimalTrack-train/DeepSORT/data/$seq_name.txt

#     cp $in_dir $out_dir
# done < <(cat $seqmap_dir/AnimalTrack-train.txt | tail -n +2)
# echo "Got mock tracker results for AnimalTrack"