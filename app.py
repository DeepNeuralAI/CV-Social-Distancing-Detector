import imutils
import cv2
from lib.utils import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", required=True,
	help="input video file")
ap.add_argument("-o", "--output", type=str, default="", required=True,
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=0,
	help="output frame should be displayed")
args = vars(ap.parse_args())



SHOW_VIDEO = args['display']
INPUT_PATH = args['input']
OUTPUT_PATH = args['output']

cap = cv2.VideoCapture(INPUT_PATH)

# Need to download cfg and weights from https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#load-the-yolo-network
net = load_coco_network(config_path='yolo-coco/yolov3.cfg', 
                        weights_path='yolo-coco/yolov3.weights')
layer_names = get_layer_names(net)
writer = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret: break
    frame = imutils.resize(frame, width = 800)
    
    results = detect(frame, network = net, layer_names = layer_names)
    violations = detect_violations(results)
    
    for index, (prob, bounding_box, centroid) in enumerate(results):
        start_x, start_y, end_x, end_y = bounding_box

        color = (0, 0, 255)  if index in violations else (0, 255, 0)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    cv2.putText(frame, f'Num Violations: {len(violations)}', (10, frame.shape[0] - 25),
		fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(0, 0, 255), thickness=1)
    
    if SHOW_VIDEO: 
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if writer is None:
        writer = video_writer(OUTPUT_PATH, frame)
    
    if writer: writer.write(frame)


cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Finished Writing Video to {OUTPUT_PATH}")
print("Cleared all windows...")
    
    
    