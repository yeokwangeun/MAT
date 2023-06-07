import argparse
from sys import platform

from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from deep_sort import DeepSort

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


#Ground truth preprocessing
def get_gt(file_path) :
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 좌표 정보를 순회하면서 이미지 패치 추출
    gt_list = []
    for line in lines:
        # 좌표 정보 파싱
        gt_list.append(list(map(int, line.strip().split(','))))

    result = {}
    for item in gt_list:
        frame_id, target_id, top_left_x, top_left_y, width, height, confidence, class_name, visibility = item
        
        if frame_id not in result:
            result[frame_id] = {}
        
        if target_id not in result[frame_id]:
            result[frame_id][target_id] = []
        
        result[frame_id][target_id] = [top_left_x, top_left_y, width, height, confidence, class_name, visibility]
    return result

def detect(save_img=True):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_img = False
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    result = []
    # 좌표 정보를 순회하면서 이미지 패치 추출
    gt_file_path = opt.gt_file
    gt_dict = get_gt(gt_file_path)
    
    #frame iteration    
    for path, img, im0s, vid_cap in dataset:
        t = time.time()
        # frame 1개 detection result extract
        for i in range(len(path)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i] #dataset+path
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            # s += '%gx%g ' % img.shape[2:]  # print string
            #frame별 object dictionary
            gt_dict_ = gt_dict[i+1]
            if gt_dict_ is not None and len(gt_dict_):
                bbox_xywh = []
                confs = []
                
                # Write results
            det = [value for value in gt_dict_.values()]
            # {"tager_id" : [top_left_x, top_left_y, width, height, confidence, class, visivility]}}
            for top_left_x, top_left_y, bbox_w, bbox_h, conf, cls, visivility in det:
                img_h, img_w, _ = im0.shape  # get image shape
                x_c = int(top_left_x+(bbox_w/2))
                y_c = int(top_left_y+(bbox_h/2))
                #print(x_c, y_c, bbox_w, bbox_h)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([conf])
                label = '%s %.2f' % (names[int(cls)], conf)
                # print('bboxes')
                # print(torch.Tensor(bbox_xywh))
                # print('confs')
                # print(torch.Tensor(confs))
                
                #deep sort update
                outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)) , im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                # print('\n\n\t\ttracked objects')
                # print(outputs)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, time.time() - t))
            result.append(np.insert(outputs, 0, i+1, axis=1))
                # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    if len(result) > 0:
        result = np.vstack(result) 
        print("Result shape: ", result.shape)
        np.savetxt(str(Path(out))+'/raw_tracker.csv', result, delimiter=",", fmt='%d')

    print('Done. (%.3fs)' % (time.time() - t0))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='yolov3/weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--gt-file', type=str, help='ground truth file path' )
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
