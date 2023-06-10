"""
For a single video, get the representative z (depth) value for each object for each frame, based on the gt file and MiDas output depth map

Usage: python3 get_z_med.py --gt <gt_file_path> --depth_dir <depth_map_folder_dir> --output <output_csv_file_path>
ex. python3 get_z_med.py --gt TrackEval/data/gt/mot_challenge/AnimalTrack-test/chicken_2/gt/gt.txt \
    --depth_dir Datasets/AnimalTrack-depth/chicken_1 --output analysis/z/chicken_1.csv

!!! depth_dir은 "각 동영상"에 대해 midas가 생성한 폴더 경로 (0001-dpt_beit_large_512.pfm 이런 파일들 들어있는 폴더)


1. read & parse gt file
    gt format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
2. Create a array (z_array) for every detected object, for every frame 
    : the initial value for each element would be an -1, the shape of the array would be (frames_num, id,)
3. for each frame, for each object, get the z value, store it in the z_array
    3.1 get the entire depth map of a frame (made with midas)
    3.2 for each object in gt file: 
        3.2.1 get the depth map of the object by cropping the frame depth map, using the bounding box info from gt
        3.2.2 get the z value of the object by averaging the depth map of the object
        3.2.3 store the z value in the z_array
4. save the z_array as csv file
"""

import numpy as np
import re 
import argparse


# Utils
def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


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


# 2. Create a array (z_array) for every detected object, for every frame 
#     : the initial value for each element would be an -1, the shape of the array would be (frames_num, id,)
def create_z_array(gt_array):
    frames_num, id_num, _ = gt_array.shape
    z_array = np.full((frames_num, id_num), -1, dtype=np.float32)
    return z_array


# 3. for each frame, for each object, get the z value, store it in the obj_depth_map array
def crop_depth_map(depth_map, coor):
    # get the depth map of the object by cropping the depth map
    # coor: gt bb_left, bb_top, bb_width, bb_height
    coor[0], coor[1] = max(coor[0], 0), max(coor[1], 0) # make coordinates non-negative
    obj_depth_map = depth_map[coor[1]:coor[1]+coor[3], coor[0]:coor[0]+coor[2]]

    return obj_depth_map # array 

def med_z(depth_map): 
    # divide the depth map into 9 parts, and take the median of the middle part
    height, width = depth_map.shape
    mid_left = int(width/3) # same as width
    mid_top = int(height/3) # same as height
    mid_depth_map = depth_map[mid_top:mid_top*2, mid_left:mid_left*2]

    mid_depth_map = np.reciprocal(mid_depth_map, dtype=np.float32, where=mid_depth_map!=0) # inverse
    return np.median(mid_depth_map)*10000 


# get the depth map for each frame and crop it for each object, then get the z value for each object, for each frame
def get_z(gt_array, depth_dir):
    z_array = create_z_array(gt_array) # array to store the z value for each object, for each frame
    frames_num, id_num, _ = gt_array.shape
    for frame in range(frames_num):
        # get the depth map (midas output) 
        try: 
            pfm = str(frame+1).zfill(4) + "-dpt_beit_large_512.pfm" 
            pfm_path = str(f"{depth_dir}/{pfm}")
            depth_map, scale = read_pfm(pfm_path) # entire image depth map. shape: (height, width)
            if scale != 1:
                print(f"Scale isn't 1 for frame {frame+1} - but I actually don't know what to do with it lol")

        except FileNotFoundError:
            print("FileNotFoundError: ", pfm_path)
            print("The depth map for this frame is not found, skipping...")
            continue

        for id in range(id_num):
            if gt_array[frame, id, 0] == -1: # if the object is not detected in this frame, skip
                continue
            else:
                coor = gt_array[frame, id, :] # (bb_left, bb_top, bb_width, bb_height)
                obj_depth_map = crop_depth_map(depth_map, coor) # object depth 
                z_array[frame, id] = med_z(obj_depth_map)
    return z_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='', help='gt txt file path, the one in TrackEval/data/gt folder')
    parser.add_argument('--depth_dir', type=str, default='', help='depth map directory for each video, NOT for the entire dataset. Containing sth like 0xxx-dpt_beit_large_512.pfm')
    parser.add_argument('--output', type=str, default='', help='output .csv file path')
    opt = parser.parse_args()

    gt_array = read_gt(opt.gt)
    z_array = get_z(gt_array, opt.depth_dir)
    np.savetxt(opt.output, z_array, delimiter=',', fmt='%.3f') # (frames_num, id)
