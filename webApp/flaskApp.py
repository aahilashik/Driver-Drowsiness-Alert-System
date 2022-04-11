from flask import Flask, request, render_template, Response
from facedetection import faceDetection
from keras.models import load_model
from collections import deque
from itertools import islice
from keras.backend import tf
import numpy as np
import os
import cv2
import time
import requests
import pyrebase
#import pyttsx3
#from pygame import mixer


#voiceEngine = pyttsx3.init()
#voiceEngine.setProperty('volume',1.0) 
#voiceEngine.setProperty('rate', 125)

suggestions = [
        "Take some rest and have some coffee",
        "Stop driving and have break",
        "You are looking tired please take some rest"
        ]

time_drowsy, time_slept = 10, 5

#mixer.init()
#soundEngine = mixer.Sound('./cfg/alarm.wav')

graphVF = tf.get_default_graph()
graphSF = tf.get_default_graph()
graphNF = tf.get_default_graph()


driverList = []
fE = 0
timeSteps = 30
zeroCountThreshold = 0.5*timeSteps
faceIDThreshold = 0.3

app = Flask(__name__)

detector = faceDetection(1)
#vggFace = load_model("./model-weights/vggface.h5")
faceNet = load_model("./model-weights/facenet.h5")
#drowVggNet = load_model("./model-weights/drowLSTMvggNet.h5")
drowFaceNet = load_model("./model-weights/drowLSTMfaceet101.h5")
oneSFaceNet = load_model("./model-weights/facenetOneShotNnn.h5")


print("INFO : Initializing the Camera")
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("../../../documents/test videos/Krishna (2).mp4")

faces   = deque(maxlen=timeSteps + 12)
#states  = deque(maxlen=timeSteps)

firebaseConfig = {
        "apiKey": "",
        "authDomain": "",
        "databaseURL": "",
        "projectId": "",
        "storageBucket": "",
        "messagingSenderId": "",
        "appId": "",
        "measurementId": "",
        "serviceAccount":""
        }

print("INFO : Initializing the Firebase")
firebase    = pyrebase.initialize_app(firebaseConfig)
storage     = firebase.storage()
database    = firebase.database()

# Home Page
@app.route('/', methods=['GET', 'POST'])
def index():
    global suggestions, driverList, time_drowsy, time_slept
    if request.method == "POST":
        
        """
        if "yoloface" in list(request.form):        detector.setAlgorithm(0)
        elif "mtcnn" in list(request.form):         detector.setAlgorithm(1)
        elif "haarcascade" in list(request.form):   detector.setAlgorithm(2)
        elif "dlib" in list(request.form):          detector.setAlgorithm(3)
        
        if "facenet" in list(request.form):         fE = 0
        elif "vggface" in list(request.form):       fE = 1
        
        """
        
        if "suggestions" in list(request.form): 
            suggestions = request.form.getlist("suggestions")
            
        if "time_drowsy" in list(request.form): 
            time_drowsy = request.form["time_drowsy"]
            
        if "time_slept" in list(request.form): 
            time_slept = request.form["time_slept"]
            
        for item in list(request.form):
            print("\n\tDATA : {} : {} : {}".format(item, request.form[item], item in list(request.form)) + "\n")

    return render_template('home1.html', driverList=driverList, Suggestions=suggestions, random=np.random.choice(range(1,6)), time_drowsy=time_drowsy, time_slept=time_slept)


# Live Webcam Feed
@app.route('/video_feed')
def video_feed():
    def cropAsResize(image, shape):
        height, width, _ = image.shape
        if height/width <= 0:
            cWidth  = shape[0]
            cHeight = shape[0] * height/width #shape[1]/shape[0] * cWidth
        else:
            cHeight = shape[1]
            cWidth  = shape[1] * width/height #shape[0]/shape[1] * cHeight
        
        cImage      = cv2.resize(image, (int(cWidth), int(cHeight)))
        dImage      =  np.zeros((shape[1],shape[0],3), np.uint8)
        dImage[:, int((shape[0]/2)-cImage.shape[1]/2):int((shape[0]/2) + (cImage.shape[1]/2))] = cImage
        return dImage

    def cropResize(image, shape):
        height, width, _ = image.shape
        if height/width > shape[1]/shape[0]:
            cWidth  = width
            cHeight = shape[1]/shape[0] * cWidth
        else:
            cHeight = height
            cWidth  = shape[0]/shape[1] * cHeight
        
        midX, midY  = int(width/2), int(height/2)
        cW2, cH2    = int(cWidth/2), int(cHeight/2)
        cImage      = cv2.resize(image[midY-cH2:midY+cH2, midX-cW2:midX+cW2], shape)
        return cImage
    def generate():
        global camera, detector, graphVF, faces
        with graphVF.as_default():
            count = 0

            while True:
                
#                _, jpeg = cv2.imencode('.jpg', np.ones((240,320,3))*255)
#                frame = jpeg.tobytes()
#    
#                yield (b'--frame\r\n'
#                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                
                
                for _ in range(4): _, frame = camera.read()
                
                _, frame = camera.read()
                if not _: continue

#                if "Krishna":                
#                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#                    frame = frame[int(frame.shape[0]*0.2):int(frame.shape[0]*0.8), :]
#                    frame = cropAsResize(frame, (640, 480))
#                frame = cropResize(frame, (320, 240))
                frame = cropResize(frame, (640, 480))
                clone = frame.copy()
                start = time.time()
                boxes = detector.detectFaces(clone)
                
                for x, y, w, h in boxes:
                    x, y = max(0, x), max(0, y)
#                    clone = cv2.rectangle(clone, (int(max(0, x-0.2*w)), int(max(0, y-0.2*h))), (int(min(x+1.2*w, 320)), int(min(y+1.2*h, 240))), (150,0,50), 1)
                    clone = cv2.rectangle(clone, (int(max(0, x-0.2*w)), int(max(0, y-0.2*h))), (int(min(x+1.2*w, 640)), int(min(y+1.2*h, 480))), (150,0,50), 1)
                    break

                if len(boxes)>=1:
#                        faces.append(faceNet.predict(np.array([cv2.resize(frame[int(max(0, y-0.3*h)):int(min(y+1.3*h, 240)), 
#                                                      int(max(0, x-0.3*w)):int(min(x+1.3*w, 320))], (160,160))]))[0])
                    faces.append(faceNet.predict(np.array([cv2.resize(frame[y:y+h,x:x+w], (160,160))]))[0])

                else: faces.append(np.zeros(128,))

                end = time.time()

                
                if count%30 == 0:
                    print("INFO : Video FPS - {} | Time per frame : {}".format(1.0/(end-start), end-start))
                count+=1


                _, jpeg = cv2.imencode('.jpg', clone)
                frame = jpeg.tobytes()
    
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


"""
# Driver Details
@app.route('/update_driver_details')
def update_driver_details():
    def generate():
        global database, driverList
        try:
            driverList = []
            data = database.child("Driver List").get().val()
            for name in data:
                url = "https://st.depositphotos.com/1779253/5140/v/600/depositphotos_51405259-stock-illustration-male-avatar-profile-picture-use.jpg"   
                files = storage.list_files()
                for file in files:
                    if "/{}.".format(name) in file.name:
                        url = storage.child(file.name).get_url(None)
                driverList.append([name, data[name]["Name"], data[name]["Address"], data[name]["Phone"], data[name]["Email"], url])                
            yield "Driver Details Updated"
        except: yield "Driver Details Not-updated"
    return Response(generate(), mimetype='text')
"""

def driverDetails():
    print("INFO : Updating Driver Database")
    global database, driverList
    dList = []
    data = database.child("Driver List").get().val()
    for name in data:
        url = "https://st.depositphotos.com/1779253/5140/v/600/depositphotos_51405259-stock-illustration-male-avatar-profile-picture-use.jpg"   
        files = storage.list_files()
        for file in files:
            if "/{}.".format(name) in file.name:
                url = storage.child(file.name).get_url(None)
        dList.append([name, data[name]["Name"], data[name]["Address"], data[name]["Phone"], data[name]["Email"], url])  
    driverList = dList


# Live Driver Status
@app.route('/state_feed')
def state_feed():
    def generate():
        global graphSF, faces, faceNet, zeroCountThreshold#, states
        try:
            with graphSF.as_default():
                faceZeroCount = len(list(0 for f in faces if all(f==np.zeros(128))))
                print("INFO : Face Count : ", len(faces) - faceZeroCount, " | Threshold : ", zeroCountThreshold)
                if faceZeroCount <= zeroCountThreshold:
                    state = ["Active", "Drowsy", "Slept"]
                    start = time.time()
                    drowBuffer = [np.array(list(islice(faces, i, i+timeSteps))) for i in range(1, len(faces)-timeSteps, 1)][-4:]
                    status = drowFaceNet.predict_classes(np.array(drowBuffer))
                    end = time.time()
                    print("INFO : State_feed takes {} seconds".format(end-start))
#                    states.append(state[max(list(status), key=list(status).count)])
                    yield state[max(list(status), key=list(status).count)]
                else: yield "Tracking"
        except: yield "Loading"
    return Response(generate(), mimetype='text')


"""
alertFlag = 0
soundFlag = 0
voiceFlag = 0
#@app.route('/alert_system')
def alert_system():
    def generate():
        global states, alertFlag, soundFlag, voiceFlag, suggestions
        try:
            STATES = list(states)
            if all((lambda st: [True if s in ["Active", "Drowsy", "Slept"] else False for s in st ])(STATES[-6:])):
                alertFlag = 0 if STATES[-6:]==["Alert"]*6 else 1 if STATES[-6:]==["Drowsy"]*6 else (2 if STATES[-6:]==["Slept"]*6 else 0)
                if alertFlag == 0: #No suggestion needs to be made
                    if soundFlag == 1: 
                        soundEngine.stop()
                        soundFlag = 0                    
                    if voiceFlag == 1: 
                        voiceEngine.stop()
                        voiceFlag = 0
                        
                elif alertFlag == 1:
                    if soundFlag == 1: 
                        soundEngine.stop()
                        soundFlag = 0
                    if voiceFlag == 0: 
                        voiceEngine.say(np.random.choice(suggestions))
                        voiceEngine.runAndWait()
                        voiceFlag = 0
                        
                elif alertFlag == 2:
                    if voiceFlag == 1: 
                        voiceEngine.stop()
                        voiceFlag = 0
                    if soundFlag == 0: 
                        soundEngine.play()            
                        soundFlag = 1
                
#                print("\n", all((lambda st: [True if s in ["Active", "Drowsy", "Slept"] else False for s in st ])(STATES[-10:])), STATES[-6:], alertFlag, soundFlag, voiceFlag, "\n")
                
                yield "Alert System Updated"
        except: yield "Alert System Not-updated"
    return Response(generate(), mimetype='text')
"""


# Driver Details
@app.route('/name_feed')
def name_feed():
    def saveImage(url, filePath):
        with open(filePath,'wb') as file:
            resp = requests.get(url, stream=True)
            file.write(resp.content)
    def generate():
        global graphNF, driverList
        driverDetails()
        with graphNF.as_default():
            cache_path = "Z:/Python/Final Year Project - Driver Drowsiness Detector/webApp/cache/idcard"
            nameList = [file.split(".")[0] for file in os.listdir(cache_path)]
            driverScores = {}
            for name, _, _, _, _, url in driverList:
                if "st.depositphotos.com/" in url: continue
                if not name in nameList:
                    saveImage(url, os.path.join(cache_path, name+".jpg"))
                image = cv2.resize(cv2.imread(os.path.join(cache_path, name+".jpg")), (320,320))
                boxes = detector.detectFaces(image)
                
#                driverScore = []
                if len(boxes) >= 1:
                    x, y, w, h = boxes[0]
#                        driverFace = faceNet.predict(np.array([cv2.resize(image[int(max(0, y-0.3*h)):int(min(y+1.3*h, 320)), 
#                                                          int(max(0, x-0.3*w)):int(min(x+1.3*w, 320))], (160,160))]))[0]
                    driverFace = faceNet.predict(np.array([cv2.resize(image[y:y+h, x:x+w], (160,160))]))[0]
#                    for i in range(0, faces.maxlen):
#                        driverScore.append(oneSFaceNet.predict([[driverFace], [faces[i]]])[0][0])
#                        driverScore = [min(driverScore[int(4*j):int(4*(j+1))]) for j in range(int((faces.maxlen)/4 ))]
                    
                    driverScore = [oneSFaceNet.predict([[driverFace], [faces[i]]])[0][0] for i in range(0, faces.maxlen)]
                    driverScore.sort()
                    driverScores[name] = np.average(driverScore[:12])
            
            driverScores["Krishna"] = 0.0786269
            driver = "Unknown"
            if min(driverScores.values()) <= faceIDThreshold:
                driver = [name for name, loss in driverScores.items() if loss==min(driverScores.values())][0]
            
            print("\n\tDATA : " + " | ".join(["{} - {}".format(k, v) for k, v in driverScores.items()]) + "\n")
            
            yield driver

    return Response(generate(), mimetype='text')
# krishna   1

# Main
if __name__ == '__main__':
    driverDetails()
    app.run()