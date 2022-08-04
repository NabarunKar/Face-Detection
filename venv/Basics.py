import cv2
import face_recognition

# Step 1:
# we are going to find the encoding of the main image and then
# use it to see if it can match it with the test image

imgMarcelo = face_recognition.load_image_file('../ImagesAttendance/Marcelo.jpg')  # loading the image
imgMarcelo = cv2.cvtColor(imgMarcelo, cv2.COLOR_BGR2RGB)  # converting to rgb so that it can be used by the library
imgTest = face_recognition.load_image_file('../ImagesAttendance/Marcelo test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Step 2: Finding the encodings

faceLoc = face_recognition.face_locations(imgMarcelo)[0]  # getting the first element of imgMarcelo
encodeMarcelo = face_recognition.face_encodings(imgMarcelo)[0]  # encoding the image we have detected

# drawing a rectangle around the face
cv2.rectangle(imgMarcelo, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255),2)  # 255,0,255 -> color = purple and 2 -> thickness
# print(faceLoc)  # prints 4 values - top, right, bottom, left


# detecting the face in the test image

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# Step 3: Comparing the 2 images and finding the distances between them
# (comparing the encodings we have obtained)

results = face_recognition.compare_faces([encodeMarcelo],encodeTest)
# print(results)  # -> prints True => Encodings are similar => They are of the same person

# But this is not enough, sometimes images of 2 different persons might return true in the
# above step. So we need to find out HOW similar the 2 images are, and for that we need to:

# Calculate the distance (lower distance = better match):
faceDis = face_recognition.face_distance([encodeMarcelo],encodeTest)
print(results,faceDis) # 0.39 for the similar images, 0.6 and false for vinicius
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# rounding off the face distance to 2 places after the decimal , followed by origin, font, scale, color = red, thickness = 2

cv2.imshow('Marcelo', imgMarcelo)
cv2.imshow('Marcelo test', imgTest)
cv2.waitKey(0)
