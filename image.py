import numpy as np
import time
import cv2
import math
import winsound

CROWD_THRESHOLD=30;

labelsPath = "./models/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "./models/yolov3.weights"
configPath = "./models/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image =cv2.imread('./data/waiting.jpg')
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("Frame Prediction Time : {:.6f} seconds".format(end - start))
boxes = []
confidences = []
classIDs = []

#div
classes = []
with open(labelsPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

peopleCount=0  #div

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5 and classID == 0:
            peopleCount+=1; #div
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
#print(peopleCount)  #div

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)

color=(255,0,0)
count=0
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in idxs:
        x, y, w, h = boxes[i]
        label = str(classes[classIDs[i]])
        if label=="person":
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(image, label, (x, y + 30), font, 1, color, 1)
            count=count+1



ind = []
for i in range(0,len(classIDs)):
    if(classIDs[i]==0):
        ind.append(i)
a = []
b = []
color = (0,255,0) 
if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            

distance=[] 
nsd = []
for i in range(0,len(a)-1):
    for k in range(1,len(a)):
        if(k==i):
            break
        else:
            x_dist = (a[k] - a[i])
            y_dist = (b[k] - b[i])
            d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
            distance.append(d)
            if(d<=100.0):
                nsd.append(i)
                nsd.append(k)
            nsd = list(dict.fromkeys(nsd))
#print(len(nsd)) #div
   
color = (0, 0, 255)
text=""
for i in nsd:
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = "Alert"
    frequency = 700  
    duration = 500  
    #winsound.Beep(frequency, duration)
    #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
           

#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
cv2.imshow("Social Distancing Detector", image)
cv2.imwrite('output.jpg', image)

#Crowd detection
print("No. of people: ",count)
if(count>CROWD_THRESHOLD):
    print("Crowd exceeding permitted number of people")
    frequency = 400  
    duration = 500
    winsound.Beep(frequency, duration)

cv2.waitKey()
