"""
Get the center x, y coordinate of every detected object, for every frame

Usage: python3 get_center_xy.py --gt_file <gt_file_path> --output_x <output_x_csv_file_path> --output_y <output_y_csv_file_path>
ex. python3 get_center_xy.py --gt_file ./TrackEval/data/gt/mot_challenge/AnimalTrack-test/chicken_2/gt/gt.txt \
    --output_x analysis/x/chicken_2_x.csv --output_y analysis/y/chicken_2_y.csv
"""

import numpy as np
import argparse

# 1. read & parse gt file
def read_gt(gt_file):
    with open(gt_file, 'r') as f:
        gt_lst = f.read().strip().split('\n')
        max_id = max([int(line.strip().split(',')[1]) for line in gt_lst])
        frames_num = gt_lst[-1].split(',')[0]
        gt_array = np.full((int(frames_num), int(max_id), 4), -1)

        for det in gt_lst:
            det = det.strip().split(',')[:6]
            frame, id, bb_left, bb_top, bb_width, bb_height = map(int, det)
            gt_array[frame-1, id-1, :] = [bb_left, bb_top, bb_width, bb_height] # gt.txt is 1-based
        
    return gt_array # (frames_num, id, 4) - zero-based!!!


# 2. Create a array for either x or y coordinate of every detected object, for every frame 
#     : the initial value for each element would be an -1, the shape of the array would be (frames_num, id, 1)
def create_x_array(gt_array):
    frames_num, id_num, _ = gt_array.shape
    array = np.full((frames_num, id_num,), -1, dtype=np.float32) 
    return array


# 3. get the center x, y value from the given coordinates
def center_xy(coor):
    # coor: bb_left, bb_top, bb_width, bb_height
    x = coor[0] + coor[2]/2
    y = coor[1] + coor[3]/2
    return x, y


def get_xy_array(gt_array):
    x_array = create_x_array(gt_array)
    y_array = create_x_array(gt_array)
    frames_num, id_num, _ = gt_array.shape
    for frame in range(frames_num):
        for id in range(id_num):
            if gt_array[frame, id, 0] == -1: # if the object is not detected in this frame, skip
                continue
            else:
                x, y = center_xy(gt_array[frame, id, :])
                x_array[frame, id] = x
                y_array[frame, id] = y

    return x_array, y_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', type=str, default='', help='gt txt file path')
    parser.add_argument('--output_x', type=str, default='', help='output x csv file path')
    parser.add_argument('--output_y', type=str, default='', help='output y csv file path')
    opt = parser.parse_args()

    gt_array = read_gt(opt.gt_file)
    x_array, y_array = get_xy_array(gt_array)
    np.savetxt(opt.output_x, x_array, delimiter=',', fmt='%d') # (frames_num, id)
    np.savetxt(opt.output_y, y_array, delimiter=',', fmt='%d') # (frames_num, id)