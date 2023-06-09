import argparse
import numpy as np
import pandas as pd
''' 
Sample key frames at 5FPS
Usage: python sample_frames.py --gt_sample <gt_sample_path> --bef_sample <bef_sample_path> --aft_sample <aft_sample_path>

'''

def sample_frames(gt_sample, bef, aft): # per video
    '''
    Input: gt, bef, aft .txt file path 

    bef (tracker output): np array with shape (#total_det, 6), for every frame 
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        ex) 1,1,763.00,272.00,189.00,38.00,1,-1,-1,-1

    aft: output sampled, following the sampling from gt
    '''
    # 1. load gt sample - list of indices to remove 
    # 2. load bef sample 
    # 3. remove indices from bef sample, save as aft sample

    # 1. load gt sample - list of indices to remove
    gt_sample = np.loadtxt(args.gt_sample) # shape (#gt_sample,)

    # 2. load bef sample
    bef = pd.read_csv(args.bef_sample, header=None) # shape (#bef_sample, 10)

    # 3. remove items in bef_sample, if its first index in a row is in gt_sample
    aft = bef[~bef.iloc[:, 0].isin(gt_sample)]
    np.savetxt(args.aft_sample, aft, delimiter=',', fmt='%d')


def main():
    sample_frames(args.gt_sample, args.bef_sample, args.aft_sample)
    # np.savetxt(args.sample_path, sampled_tracker, delimiter=',', fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tracker output to MOTchallenge format')
    parser.add_argument('--gt_sample', type=str, help='gt sample path')
    parser.add_argument('--bef_sample', type=str, help='bef sample path')
    parser.add_argument('--aft_sample', type=str, help='path to save sampled tracker output')
    args = parser.parse_args()

    main()