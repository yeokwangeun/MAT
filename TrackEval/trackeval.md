# How to run trackers & evaluate using TrackEval
TrackEval repo: https://github.com/JonathonLuiten/TrackEval

## 0. Setup
- Install all the requirements in `TrackEval/requirements.txt` (`pip3 -r install requirements.txt`) 
- Make sure you have all the ground-truth data ready in `TrackEval/data/gt`

## 1. Run tracker & convert tracker outputs to required format
```
bash <path-to-TrackEval>/scripts/run_tracker.sh <video source folder> <output folder> <model name> <mode: train/test> 
ex. bash ./MAT/TrackEval/scripts/run_tracker.sh /home/n0/usr/AnimalTrack/test /home/n0/usr/AnimalTrack/result deepsort train
```
### Before running the script
Adapt to your your own paths
- MAT (DeepSORT) directory
- Yolo cfg, weights

### What it does
- Run tracker
    - Run tracker on all videos in the source folder
    - Tracker outputs are stored in the output folder
- Convert tracker outputs 
    - Convert to MOTChallenge format using `TrackEval/utils/convert_tracker_format.py`:
    - Outputs are stored under `TrackEval/data/trackers/mot_challenge/AnimalTrack-<test/train>/<model>/data/<vid-name>.txt`
        ex. `TrackEval/data/trackers/mot_challenge/AnimalTrack-test/DeepSORT/data/chicken_1.txt`

## 2. Run TrackEval
``` 
# Usage
python ./MAT/TrackEval/scripts/run_mot_challenge.py --BENCHMARK AnimalTrack --SPLIT_TO_EVAL <train/test> --USE_PARALLEL False --DO_PREPROC False

# ex. 
python ./MAT/TrackEval/scripts/run_mot_challenge.py --BENCHMARK AnimalTrack --SPLIT_TO_EVAL train --USE_PARALLEL False --DO_PREPROC False 
```
- more options available in `TrackEval/scripts/run_mot_challenge.py`

### What it does
- evaluate the tracker output in `TrackEval/data/trackers/mot_challenge/AnimalTrack-<test/train>/<tracker-model>/data` 
- can set the evaluation mode to train or test. The sequence names are specified in `TrackEval/data/gt/mot_challenge/seqmaps`
- if not specified, will run evaluation for all trackers (ex. DeepSORT)
- save the results for each tracker in tracker data folder (`TrackEval/data/trackers/mot_challenge/AnimalTrack-<test/train>/<tracker-model>`)