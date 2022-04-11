from mtcnn import MTCNN
from yoloface import yoloFace
from dlib import get_frontal_face_detector
import cv2
import time


class faceDetection:
    
    def __init__(self, algorithm = 2):
        self.algorithms = ["yoloFace", "mtcnn", "haarcascade", "dlib"]
        self.algorithm = self.algorithms[algorithm] if type(algorithm)==int else algorithm
        self.detectFunction = {
                "yoloFace"      :   self.detectYoloFace,
                "mtcnn"         :   self.detectMTCNN,
                "haarcascade"   :   self.detectHaarCascade,
                "dlib"          :   self.detectDlib
                }
        self.detector = {
                "yoloFace"      :   yoloFace(),
                "mtcnn"         :   MTCNN(),
                "haarcascade"   :   cv2.CascadeClassifier('model-weights/haarcascade_frontalface_default.xml'),
                "dlib"          :   get_frontal_face_detector(),
                }
            

    def setAlgorithm(self, algorithm):
        self.algorithm = self.algorithms[algorithm] if type(algorithm)==int else algorithm
        
        
    def detectYoloFace(self, frame):
        return self.detector[self.algorithm].detectFaces(frame, 0)
        

    def detectMTCNN(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = self.detector[self.algorithm].detect_faces(frame)
        return [info["box"] for info in data]
    
    
    def detectHaarCascade(self, frame):
        return self.detector[self.algorithm].detectMultiScale(frame)

    
    def detectDlib(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector[self.algorithm](gray, 1)
        return list([r.left(), r.top(), r.right()-r.left(), r.bottom()-r.top()] for r in rects)


    def detectFaces(self, frame):
        return self.detectFunction[self.algorithm](frame)
    
    

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    detector = faceDetection()
    detector.setAlgorithm("dlib")
    
    while True:
        _, frame = cap.read()
        
        start = time.time()
        boxes = detector.detectFaces(frame)
        print("It takes {} seconds | FPS : {}".format(time.time()-start, 1.0/(time.time()-start)))
    
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (143, 30, 13), 2)
            
        cv2.imshow("Face", frame)
        if cv2.waitKey(1) == ord("q"): break
    
    cv2.destroyAllWindows()
    cap.release() 
    






