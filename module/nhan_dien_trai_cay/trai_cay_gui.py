import cv2 as cv
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Cấu hình model và các tham số
model = os.path.join(current_dir, 'yolov8_nhan_dien_trai_cay.onnx')
filename_classes = os.path.join(current_dir, 'trai_cay.txt')
mywidth = 640
myheight = 640
postprocessing = 'yolov8'
background_label_id = -1
backend = 0
target = 0

# Load names of classes
classes = None
if filename_classes:
    with open(filename_classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

# Load a network
net = cv.dnn.readNet(model)
net.setPreferableBackend(0)
net.setPreferableTarget(0)
outNames = net.getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
scale = 0.00392
mean = [0, 0, 0]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=3)

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        font_scale = 0.8
        font_thickness = 1
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        top = max(top, labelSize[1])

        # Giảm nền dư xung quanh nhãn
        label_ymin = max(top - labelSize[1], 0)
        label_ymax = min(top + baseLine, frameHeight)
        label_xmax = min(left + labelSize[0], frameWidth)

        cv.rectangle(frame, (left, label_ymin), (label_xmax, label_ymax), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return frame

def detect(image_array):
    frame = cv.cvtColor(image_array, cv.COLOR_RGB2BGR)
    inpWidth = mywidth if mywidth else frame.shape[1]
    inpHeight = myheight if myheight else frame.shape[0]
    blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv.CV_8U)

    net.setInput(blob, scalefactor=scale, mean=mean)
    outs = net.forward(outNames)
    processed_frame = postprocess(frame, outs)
    
    # Chuyển đổi ảnh đã xử lý từ BGR sang RGB
    processed_img = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
    return processed_img