import cv2
import numpy as np


#set of predefined upper and lower bound average differences in neighboring control points (local extrema)
fistL = 24
fistU = 109
palmL = 113
palmU = 197
splayL = 280

#coordinates for locations
upend = 882
lowstart = 1763
leftend = 840
rightstart = 1680

def classify(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    #rotating image since I oriented myself from the side 
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #learned how to threshold on a specific color using: https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    #learned gaussian blur to reduce noise using: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #learned to threshold grayscale to get binary image using: https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
    ret,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)

    #learned contours using: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #learned how to approximate contours using: https://docs.opencv.org/3.4/dc/dcf/tutorial_py_contour_features.html and https://www.tutorialspoint.com/how-to-approximate-a-contour-shape-in-an-image-using-opencv-python
    #approximated contours because I wanted a simpler outline of the hand to work with that still displayed the gesture but not all the minute details
    for c in contours:
        accuracy= 0.001 * cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c,accuracy,True)
        cv2.drawContours(img, [approx], 0, (0,255,0),2) 

    #learned about centroids and moments using: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    #using the center of the contours I generated to determine the location of the hand most accurately as people associate their hand location with the center of the hand not the edges
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

    #if statements to determine location based of center of hand
    if cx < leftend:
        if cy < upend:
            location = "Upper Left"
        elif cy > lowstart:
            location = "Lower Left"
        else:
            location = "None"
    elif cx > rightstart:
        if cy < upend:
            location = "Upper Right"
        elif cy > lowstart:
            location = "Lower Right"
        else:
            location = "None"
    elif leftend <= cx <= rightstart:
        if upend<=cy<=lowstart:
            location = "Center"
        else:
            location = "None"

    #finding local extrema by looping through the approximated contours to find points where both neighbors or either less than or greater than the current point
    #this determines where there is a peak or valley in the outline of the hand
    cp=[]
    for i in range(1, len(approx)-1):
        curr = approx[i][0]
        neighbor1 = approx[i-1][0]
        neighbor2 = approx[i+1][0]
        if curr[1] > neighbor1[1] and curr[1] > neighbor2[1]:
            cv2.circle(img, (curr[0], curr[1]), 10, (0, 0, 255), -1)
            cp.append(curr)
        elif curr[1] < neighbor1[1] and curr[1] < neighbor2[1]:
            cv2.circle(img, (curr[0], curr[1]), 10, (255, 0, 0), -1)
            cp.append(curr)

    #finding average difference in neighboring local extrema to determine how extreme the variations in the hand are, this helps identify the differences between the different gestures
    diff=0
    for i in range(len(cp)-1):
        curr= cp[i]
        neighbor = cp[i+1]
        diff = diff + np.absolute(curr[1]-neighbor[1])
    avgdiff = diff/(len(cp)-1)

    #if statements to determine the hand gesture and put text on the image to display the gesture and location
    if fistL <= avgdiff <= fistU:
        gesture = "Fist"
    elif palmL < avgdiff <= palmU:
        gesture = "Palm"
    elif splayL < avgdiff:
        gesture = "Splay"
    else:
        gesture = "Unknown"

    img = cv2.putText(thresh, gesture + ", " + location, (1000,2500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 10, cv2.LINE_AA)

    cv2.imshow("image",img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return (gesture,location)

#easy sequence my images
img1 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/EzLock1.jpeg"
img2 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/EzLock2.jpeg"
img3 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/EzLock3.jpeg"
c1 = classify(img1)
c2 = classify(img2)
c3 = classify(img3)

if c1[0]=="Fist":
    if c2[0]=="Splay" and c2[1]=="Upper Right":
        if c3[1]=="Lower Right":
            print("Unlocked")
        else:
            print("Failed")
    else:
        print("Failed")
else:
    print("Failed")



#hard sequence my images
img1 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/HardLock1.jpeg"
img2 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/HardLock2.jpeg"
img3 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/HardLock3.jpeg"
c1 = classify(img1)
c2 = classify(img2)
c3 = classify(img3)

if c1[0]=="Palm" and c1[1]=="Lower Left":
    if c2[0]=="Unknown" and c2[1]=="Center":
        if c3[0]=="Splay" and c3[1]=="None":
            print("Unlocked")
        else:
            print("Failed")
    else:
        print("Failed")
else:
    print("Failed")


#interesting sequence my images
img1 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/Interesting1.jpeg"
img2 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/Interesting2.jpeg"
img3 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/Interesting3.jpeg"
c1 = classify(img1)
c2 = classify(img2)
c3 = classify(img3)

if c1[0]!="Splay":
    if c2[0]=="Fist" and c2[1]=="None":
        if c3[0]=="Palm" and c3[1]=="Center":
            print("Unlocked")
        else:
            print("Failed")
    else:
        print("Failed")
else:
    print("Failed")

#easy sequence my friend's images
img1 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/EzGuest1.jpeg"
img2 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/EzGuest2.jpeg"
img3 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/EzGuest3.jpeg"
c1 = classify(img1)
c2 = classify(img2)
c3 = classify(img3)

if c1[0]=="Fist":
    if c2[0]=="Splay" and c2[1]=="Upper Right":
        if c3[1]=="Lower Right":
            print("Unlocked")
        else:
            print("Failed")
    else:
        print("Failed")
else:
    print("Failed")



#hard sequence my friend's images
img1 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/HardGuest1.jpeg"
img2 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/HardGuest2.jpeg"
img3 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/HardGuest3.jpeg"
c1 = classify(img1)
c2 = classify(img2)
c3 = classify(img3)

if c1[0]=="Palm" and c1[1]=="Lower Left":
    if c2[0]=="Unknown" and c2[1]=="Center":
        if c3[0]=="Splay" and c3[1]=="None":
            print("Unlocked")
        else:
            print("Failed")
    else:
        print("Failed")
else:
    print("Failed")


#interesting sequence my friend's images
img1 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/IntGuest1.jpeg"
img2 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/IntGuest2.jpeg"
img3 = "/Users/tylerberg/Desktop/Visual Interfaces Assignment 1/images/IntGuest3.jpeg"
c1 = classify(img1)
c2 = classify(img2)
c3 = classify(img3)

if c1[0]!="Splay":
    if c2[0]=="Fist" and c2[1]=="None":
        if c3[0]=="Palm" and c3[1]=="Center":
            print("Unlocked")
        else:
            print("Failed")
    else:
        print("Failed")
else:
    print("Failed")


