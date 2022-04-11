import time
import numpy as np
import cv2

"""
Get the Weight File 'model-weights/yolov3-wider_16000.weights' : 
    Download from the below link :
        https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view?usp=sharing

"""
class yoloFace():
    def __init__(self, conf_threshold=0.7, nms_threshold=0.4, 
                 model_cfg='cfg/yolov3-face.cfg',
                 model_weights='model-weights/yolov3-wider_16000.weights',
                 preferableBackend=cv2.dnn.DNN_BACKEND_OPENCV, 
                 preferableTarget=cv2.dnn.DNN_TARGET_CPU):
        
        self.CONF_THRESH   = conf_threshold
        self.NMS_THRESH    = nms_threshold
        self.IMG_WIDTH     = 416
        self.IMG_HEIGHT    = 416
        self.model_cfg     = model_cfg
        self.model_weights = model_weights
        
        self.net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_weights)
        self.net.setPreferableBackend(preferableBackend)
        self.net.setPreferableTarget(preferableTarget)


    # Get the names of the output layers
    def get_outputs_names(self, net):
        layers_names = net.getLayerNames()
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    
    def processing(self, frame, outs, conf_thresh, nms_thresh):
        frame_height = frame.shape[0]
        frame_width_ = frame.shape[1]
        
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs:
            for detection in out:
                scores     = detection[5:]
                class_id   = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_thresh:
                    width  = int(detection[2] * frame_width_)
                    height = int(detection[3] * frame_height)
                    left   = int(int(detection[0] * frame_width_) - width  / 2)
                    top    = int(int(detection[1] * frame_height) - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

        for i in indices:
            box = boxes[i[0]]
            final_boxes.append(box)
        return final_boxes
    
    
    # Detect Faces in frame and return the boundary box points
    def detectFaces(self, frame, showElapsedTime=1):
        start = time.time()
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.IMG_WIDTH, 
                                    self.IMG_HEIGHT), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_outputs_names(self.net))
        faces = self.processing(frame, outs, self.CONF_THRESH, self.NMS_THRESH)
        if showElapsedTime: print("It takes {} seconds | FPS : {}".format(time.time()-start, 1.0/(time.time()-start)))
        return faces
