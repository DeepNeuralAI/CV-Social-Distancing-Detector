import imutils
import cv2
from lib.utils import *

SHOW_VIDEO = False
VIDEO_PATH = 'video_files/walking.mp4'
OUTPUT_VIDEO_PATH = 'video_files/output.avi'

cap = cv2.VideoCapture(VIDEO_PATH)

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
		fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.85, color=(0, 0, 255), thickness=1)
    
    if SHOW_VIDEO: 
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if writer is None:
        writer = video_writer(OUTPUT_VIDEO_PATH, frame)
    
    if writer: writer.write(frame)

print("Finished Writing Video")
cap.release()
writer.release()
cv2.destroyAllWindows()
print("Cleared all windows...")
    
    
    