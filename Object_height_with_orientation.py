'''
Created on 11-Mar-2019

@author: dhanalaxmi
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from operator import itemgetter
from imutils import perspective

# load the image, convert it to grayscale, and segment reference object using thresholding techniques

image = cv2.imread("/home/dhanalaxmi/workspace/AI/Object_Height/1.jpeg")
img=cv2.imread("/home/dhanalaxmi/workspace/AI/Object_Height/2.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
thresh = cv2.threshold(gray, 100,220, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

#Find contours(bounding box) of reference object
#Determine pixel height of Reference object i an image

cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
count=0
for c in cnts: 
    area=cv2.contourArea(c)
    if area>130000 and area<150000:
        
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), 3)
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.show()
object_corner=box

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
thresh = cv2.threshold(gray, 100,220, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
      
cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
count=0
for c in cnts: 
    area=cv2.contourArea(c)
    if area>85000 and area<100000:
#         cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.099* epsilon, True)
#         print(area,len(approx))
 
        if len(approx)==4:
            tl = tuple(c[c[:, :, 0].argmin()][0])
            br = tuple(c[c[:, :, 0].argmax()][0])
            tr = tuple(c[c[:, :, 1].argmin()][0])
            bl = tuple(c[c[:, :, 1].argmax()][0])
#             cv2.circle(img, tl, 8, (0, 0, 255), -1)
#             cv2.circle(img, br, 8, (0, 255, 0), -1)
#             cv2.circle(img, tr, 8, (255, 0, 0), -1)
#             cv2.circle(img, bl, 8, (255, 255, 0), -1)
                 
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.show()
 
obj_corner=(tl,tr,br,bl)
obj_corner=(np.asarray(obj_corner, dtype="float32"))


h, mask = cv2.findHomography(obj_corner,object_corner)
      
out = cv2.warpPerspective(img, h, (1500,1000))

# plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
# plt.show()

gray = cv2.cvtColor(out, cv2.COLOR_BGRA2GRAY)
thresh = cv2.threshold(gray, 100,220, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
       
cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
count=0
for c in cnts: 
    area=cv2.contourArea(c)
    if area>130000 and area<150000:
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(out, [box.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), 3)
 

RO_Height=box[3][1]-box[0][1]
print("RO pixel is {}".format(RO_Height))  
######### Implement SIFT Detector ########
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(thresh, None)
# Convert the Key Points into an array of coordinates
points = []
for point in kp:
    
    points.append(point.pt)
# Determine the Top Most Points of Bottle
points_ordered = sorted(points, key=itemgetter(1), reverse=False)
head = [points_ordered[0]]

cv2.circle(out, (int(head[0][0]), int(head[0][1])), 2, (0, 0, 255), 8)

# Determine the Bottom Most Points of Bottle

gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

minLineLength=500
lines = cv2.HoughLinesP(image=edges,rho=.7,theta=np.pi/180, threshold=50,lines=np.array([]), minLineLength=minLineLength,maxLineGap=500)
bl_listing=[]
for i in range(len(lines)):
    if lines[i][0][1] >620:
        if lines[i][0][3] >650:
            bl_listing.append(lines[i][0])

points_ordered = sorted(bl_listing, key=itemgetter(1), reverse=True)
bottom=points_ordered[0]
a,b,c,d=bottom
midpoint=((a + c) * 0.5), ((b + d) * 0.5)
cv2.line(out,(a,b),(c,d), (0, 0, 255), 3, cv2.LINE_AA)

Bottle_Height=(midpoint[1]-head[0][1]) 
print("Bottle pixel is {}".format(Bottle_Height))  

RO_CM =14.8
# Calculate the conversion factor
CM_per_Pixel = (RO_CM/ RO_Height)

# Calculate the Kid's Height in cm
Bottle_Height_CM = (Bottle_Height * CM_per_Pixel)
print("Bottle Height in cm -  {}".format(Bottle_Height_CM))  

plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
plt.show()