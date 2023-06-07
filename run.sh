#!/bin/bash

ANIMALS=("chicken" "deer" "dolphin" "duck" "goose" "horse" "penguin" "pig" "rabbit" "zebra")
BASEDIR="/home/kwangeunyeo/MAT"
DATADIR="/home/kwangeunyeo/animaltrack/test_videos"
OUTDIR="${BASEDIR}/output_base_finetune"

for CLASS in "${!ANIMALS[@]}"; do
  ANIMAL="${ANIMALS[${CLASS}]}"
  python ${BASEDIR}/track.py --yolo-weights ${BASEDIR}/yolov5/weights/animaltrack_finetune.pt --cnn-weights ${BASEDIR}/deep_sort/deep/checkpoint/base.t7 --source ${DATADIR}/${ANIMAL} --output ${OUTDIR}/${ANIMAL} --classes ${CLASS} --save-txt --half
done