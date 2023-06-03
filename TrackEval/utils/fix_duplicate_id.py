# --------------------------------------------------------
# This script works only when the same id repeats twice within one frame.
# If the same id repeats more than twice within one frame, don't worry,
# simply replace the original gt folders with generated gt_checked folders,
# and run the script again, repeat above procedure until no error log shows up.
# --------------------------------------------------------
import os

root = r"/home/n0/mlvu014/MAT/MAT_test"
classes = ['chicken', 'deer', 'dolphin', 'duck', 'goose', 'horse', 'penguin', 'pig', 'rabbit', 'zebra', 'whole']
processed = []

for cl in classes:
    if cl == 'whole':
        # original ground truth files
        gt_paths = os.path.join(root, "Whole AnimalTrack", "gt_all")
        if not os.path.exists(gt_paths):
            continue
        # where corrected ground truth files will be stored
        gt_dst_paths = os.path.join(root, "Whole AnimalTrack", "gt_all_checked")
        os.makedirs(gt_dst_paths, exist_ok=True)
    else:
        gt_paths = os.path.join(root, cl, 'gt')  # i.e. D:\AnimalTrack\chicken\gt
        if not os.path.exists(gt_paths):
            continue
        print("gt_paths:", gt_paths)
        gt_dst_paths = os.path.join(root, cl, 'gt_checked')  # i.e. D:\AnimalTrack\chicken\gt_checked
        os.makedirs(gt_dst_paths, exist_ok=True)

    processed.append(gt_paths)
    for gt_file in os.listdir(gt_paths):
        gt_path = os.path.join(gt_paths, gt_file)
        dst_path = os.path.join(gt_dst_paths, gt_file)
        f = open(gt_path, "r")
        f_o = open(dst_path, "w")
        lines = f.readlines()
        duplicate_ids = []
        info_dict = {}

        # new ids for previous duplicate ones will start from max_id
        max_id = -1
        for line in lines:
            num_list = line.split(',')
            obj_id = int(num_list[1])
            if obj_id > max_id:
                max_id = obj_id
        max_id += 1

        # check if duplicate id exists
        for line in lines:
            num_list = line.split(',')
            frame_id = num_list[0]
            obj_id = int(num_list[1])
            if frame_id not in info_dict.keys():
                info_dict[frame_id] = [obj_id]
            elif obj_id not in info_dict[frame_id]:
                info_dict[frame_id].append(obj_id)
            # found a duplicate id within current frame
            else:
                if obj_id not in duplicate_ids:
                    duplicate_ids.append(obj_id)
                new_id = str(max_id + duplicate_ids.index(obj_id))
                num_list[1] = new_id
                # print the error log, if nothing is printed, your dataset has no error
                print("Fixed an Error! Duplicate id in {}, frame: {}, previous_id: {}, new id: {}".format(gt_file, frame_id, obj_id, new_id))

            # save corrected content (even when no duplicate id was found)
            newLine = ','.join(x for x in num_list)
            f_o.write(newLine)

        f.close()
        f_o.close()

# In case your data path is wrong, no error log doesn't mean no error then.
print("{} folders examined:".format(len(processed)))
for path in processed:
    print(path)
