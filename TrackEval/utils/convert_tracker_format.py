import argparse
import numpy as np

def convert_tracker_format(dets): # per video
    '''
    Input (tracker output): np array with shape (#total_det, 6) 
        <frame>, <bb_left>, <bb_top>, <bb_right>, <bb_bottom>, <id>

    Required format: np array with shape (#total_det, 10)
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        ex) 1,1,763.00,272.00,189.00,38.00,1,-1,-1,-1
    '''
    # Convert bottom right coordinates to width/height
    dets[:, 3] = np.abs(dets[:, 3] - dets[:, 1]) # width
    dets[:, 4] = np.abs(dets[:, 4] - dets[:, 2]) # height

    # move track_id to the 2nd col (idx 1)
    dets = dets[:, [0, 5, 1, 2, 3, 4]]

    # fill conf, x, y, z with dummy values (-1) 
    dets = np.concatenate((dets, np.ones((dets.shape[0], 4))*-1), axis=1)
    return dets


def main():
    raw_tracker = np.loadtxt(args.raw_tracker_path, delimiter=',')
    converted_tracker = convert_tracker_format(raw_tracker)
    np.savetxt(args.convert_path, converted_tracker, delimiter=',', fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tracker output to MOTchallenge format')
    parser.add_argument('--raw_tracker_path', type=str, help='path to raw tracker output file')
    parser.add_argument('--convert_path', type=str, help='path to save converted tracker output')
    args = parser.parse_args()

    main()