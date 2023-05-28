import argparse
import numpy as np

def convert_gt_format(dets):
    '''
    Input (gt) format: (#det, 9) 
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

    Required format: (#det, 10)
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        ex) 1,1,763.00,272.00,189.00,38.00,1,-1,-1,-1
    '''
    dets = np.concatenate((dets[:, :7], np.ones((dets.shape[0], 3))*-1), axis=1)
    return dets


def main():
    raw_gt = np.loadtxt(args.raw_gt_path, delimiter=',')
    converted_gt = convert_gt_format(raw_gt)
    np.savetxt(args.convert_path, converted_gt, delimiter=',', fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert gt to MOTchallenge format')
    parser.add_argument('--raw_gt_path', type=str, help='path to raw gt file')
    parser.add_argument('--convert_path', type=str, help='path to save converted gt')
    args = parser.parse_args()

    main()