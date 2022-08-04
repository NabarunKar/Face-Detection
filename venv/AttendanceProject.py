import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# Step 1: Import the images and convert them into rgb
# Let us create a list that will generate encodings automatically for all
# the images in our folder

path = '../ImagesAttendance'
images = []
classNames = []

# grab the list of images from the folder
myList = os.listdir(path)
# print(myList) # prints the names of the files in the dir

# Step 2:
# now we are going to use these names and import these images one by one
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # excluding the extensions
print(classNames)


# Step 3: Find the encodings
def findEncodings(images):
    encodeList = []  # list which stores the encodings
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting to rgb
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)  # List of known faces
print('Encoding Complete')

# Final Step: Mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:  # r+ -> read and write at the same time
        # We need to ensure someone who has already arrived doesn't get marked twice
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',') # Now we have separated the comma separated values in the csv file
            nameList.append(entry[0]) # entry[0] -> name
        # now we need to check whether the name passed into the function is already present or not
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S') # string format
            f.writelines(f'\n{name},{dtString}') # writing the name and current time in a new line


# Step 4: Match image from webcam with these encodings

# Initializing the webcam:
cap = cv2.VideoCapture(0)

# while loop to get each frame one by one
while True:
    success, img = cap.read()  # this will give us our image
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing our images so that real time processing becomes easier
    # (0,0) -> we don't want to define any pixel size so None
    # 0.25, 0.25 -> scale => 1/4th of the original
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # converting to RGB

    # there might be multiple faces in the webcam, so we are going to find the locations of all those
    # faces and pass the locations to the encoding function
    facesCurFrame = face_recognition.face_locations(
        imgS)  # faces in our current frame = all locations in our small image
    # encodings in current frame = small image, locations of the faces
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Finding the matches
    # Iterate through all the faces we found in the current frame
    # And then match it with our encodings obtained before
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  # zip since we need both lists in the same loop
        # Doing the matching
        matches = face_recognition.compare_faces(encodeListKnown,
                                                 encodeFace)  # Comparing list of known faces with one of the encodings (encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Finding the face distance
        # print(faceDis)  # we'll be able to see distance of webcam face with the faces in our images directory in the
        # console
        matchIndex = np.argmin(faceDis) # index of the image which has least distance with webcam image
        # Now we know which player has arrived
        # Display a box around that person and write their name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)  # prints the name of the person whose face is least farthest from webcam face
            y1, x2, y2, x1 = faceLoc # this comes from facesCurFrame
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 # multiplying by 4 since we resized it down to 1/4th
            # Drawing a rectangle around the original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


