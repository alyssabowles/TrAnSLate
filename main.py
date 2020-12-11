"""
A simple program that performs real-time classification of ASL letters with openCV
"""
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Start the video capture
cap = cv2.VideoCapture(0)

# IMPORTANT: standardize the images we are getting in
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Load the model
model = load_model('asl_final_model.h5'.format(9575))

# Setting up the input image size and frame crop size.
imsize = 200
csize = 400

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', 'del', 'nothing', 'space']

blue_color = (255, 0, 0)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
white_color = (255, 255, 255)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Setting size of frame
    frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # The image within this rectangle is what will be classified
    cv2.rectangle(frame, (0, 0), (csize, csize), red_color, 3)

    img = frame[0:csize, 0:csize]
    img = cv2.resize(img, (imsize, imsize))
    img = (np.array(img)).reshape((1, imsize, imsize, 3))
    img = data_generator.standardize(np.float64(img))

    # Input image into model to make prediction
    prediction = np.array(model.predict(img))
    pred = letters[prediction.argmax()]

    # Get confidence score in percentage
    confidence_score = prediction[0, prediction.argmax()] * 100
    if confidence_score > 65:
        cv2.putText(frame, 'High Confidence: {} - {:.3f}%'.format(pred, confidence_score), (10, 450), 1, 2, green_color, 2, cv2.LINE_AA)
    elif 20 < confidence_score <= 65:
        cv2.putText(frame, 'Low Confidence: {} - {:.3f}%'.format(pred, confidence_score), (10, 450), 1, 2, red_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, letters[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

    # Display the image with prediction
    cv2.imshow('frame', frame)

    # To quit the program, press q
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

# Exiting
cap.release()
cv2.destroyAllWindows()