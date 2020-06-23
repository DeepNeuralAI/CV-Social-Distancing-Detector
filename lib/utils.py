from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2


def load_coco_network(config_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net

def get_layer_names(net):
    # YOLO CNN output layer names
    #['yolo_82', 'yolo_94', 'yolo_106']
    layer_names = net.getLayerNames()
    ln = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return ln

def detect(frame, network, layer_names):
    # https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    results, boxes, confidences, classIDs, centroids = [],[], [], [], []
    h,w = frame.shape[:2]

    # Pre-processing -- mean subtraction and scaling
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    layer_outputs = network.forward(layer_names)
    
    for layer in layer_outputs:
        for detection in layer:
            '''
            The outputs object are vectors of lenght 85
            4x the bounding box (centerx, centery, width, height)
            1x box confidence
            80x class confidence
            '''
            cx, cy, bw, bh = detection[:4]
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 and classID == 0:
                box = [cx, cy, bw, bh] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # get top-left corner of image/frame
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                
                boxes.append(box)
                confidences.append(float(confidence))
                centroids.append((centerX, centerY))
                classIDs.append(classID)
    
    indices = apply_non_max_supression(boxes, confidences)
    results = create_result_tuple(boxes, confidences, centroids, indices, results)
    return results

def detect_violations(results):
    violations = set()
    if len(results) >= 2:
        
        centroids = np.array([r[2] for r in results])
        distance_matrix = dist.cdist(XA=centroids, XB=centroids, metric='euclidean')

        for row in range(distance_matrix.shape[0]):
            for col in range(row + 1, distance_matrix.shape[1]):
                if distance_matrix[row, col] < 50:
                    violations.add(row)
                    violations.add(col)
    return violations


def video_writer(output_file, frame):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('output.avi', fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    return writer

def create_result_tuple(boxes, confidences, centroids, indices, results):
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + w), centroids[i])
            results.append(r)
    return results


def apply_non_max_supression(boxes, confidences):
    return cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
