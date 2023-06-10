import pandas as pd
import numpy as np
import cv2

# animal list: chicken[1,2], deer[1-3], dolphin[1-3], duck[1-3], goose[1-3], horse[1-3], penguin[1-3], pig[1-2], rabbit[1-2], zebra[1-2]
animal = 'deer'
vid = 2 # video number
obj = [0,1,2,3,6] # object to track 
start_frame = 0
stop_frame = 306 # frame to draw the track on
coor_dir = "./MAT/analysis/coordinates" # gt x,y coordinates dir
out_dir = "./MAT/analysis/figures" 
vid_dir = "./Datasets/AnimalTrack/test" # path to folder with AnimalTrack test set videos 


def get_track(animal, vid, obj_lst, start_frame, stop_frame):
    x = pd.read_csv(f"{coor_dir}/x/x_{animal}_{vid}.csv", header=None)
    y = pd.read_csv(f"{coor_dir}/y/y_{animal}_{vid}.csv", header=None)
    trace_lst = []

    for obj in obj_lst:
        obj_x = x.iloc[start_frame:stop_frame+1, obj]
        obj_y = y.iloc[start_frame:stop_frame+1, obj]

        # delete all the rows with -1
        obj_x = obj_x[obj_x != -1]
        obj_y = obj_y[obj_y != -1]

        trace = pd.concat([obj_x, obj_y], axis=1).to_numpy()
        trace_lst.append(trace)

    return trace_lst


# draw a line connecting two points on an image
def draw_line(image, start_point, end_point, color, thickness):
    return cv2.line(image, start_point, end_point, color, thickness)


# draw the combined track until the designated frame of a video
def draw_combined_track(trace_list, video_path, stop_frame, output_path=None):

    # get the frame to draw the track on 
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    last_frame = frames[stop_frame] 
    
    # for each animal
    for trace_data in trace_list: 
        # random color generation
        color = np.random.randint(0, 255, size=3).tolist()
        for frame_number, (x, y) in enumerate(trace_data):
            x, y = int(x), int(y)

            # draw a trace line, connecting the current position with the previous position
            if frame_number > 0:
                prev_x, prev_y = int(trace_data[frame_number-1][0]), int(trace_data[frame_number-1][1])
                last_frame = draw_line(last_frame, (prev_x, prev_y), (x, y), color, 10)

        cv2.circle(last_frame, (x, y), 20, color, -1) # last point

    # save the output image with the combined track
    output_path = output_path
    cv2.imwrite(output_path, last_frame)
    print(f"Output image saved at: {output_path}")
    cap.release()


trace_lst = get_track(animal, vid, obj, start_frame, stop_frame)
video_path = f'{vid_dir}/{animal}_{vid}.mp4'
draw_combined_track(trace_lst, video_path, stop_frame, output_path=f"{out_dir}/{animal}_{vid}_{obj}.jpg")
