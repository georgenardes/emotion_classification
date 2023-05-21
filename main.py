import mediapipe as mp
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import time
import numpy as np


# Initialize face detector
mp_face_detection = mp.solutions.face_detection
facedetection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)

# Initialize emotion classifier
em_classifier = tf.keras.models.load_model("adapted_best.h5", custom_objects={"loss":tfa.losses.SigmoidFocalCrossEntropy})
print(em_classifier.summary())
class_names = ["Negative", "Neutral", "Positive"]
print(class_names)
time.sleep(1)


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:

      
      for detection in results.detections:

        # here we need to crop the image and predict the emotion
        
        # img size
        im_h = image.shape[0]
        im_w = image.shape[1]
        

        #bbox relative to pix
        bbox_xi = int(detection.location_data.relative_bounding_box.xmin * im_w)
        bbox_yi = int(detection.location_data.relative_bounding_box.ymin * im_h)
        bbox_xf = int(bbox_xi + (detection.location_data.relative_bounding_box.width * im_w))
        bbox_yf = int(bbox_yi + (detection.location_data.relative_bounding_box.height * im_h))
        # print(bbox_xi, bbox_yi, bbox_xf, bbox_yf)

        # add offset
        bbox_xi = max(min(bbox_xi - 5, im_w-1), 0)
        bbox_yi = max(min(bbox_yi - 10, im_h-1), 0) 
        bbox_xf = max(min(bbox_xf + 5, im_w-1), 0)
        bbox_yf = max(min(bbox_yf + 10, im_h-1), 0)

        # clip for boundaries


        # get the face croped
        croped_face_img = image[bbox_yi:bbox_yf, bbox_xi:bbox_xf]

        # resize the image
        croped_face_img = cv2.cvtColor(cv2.resize(croped_face_img, (48,48)), cv2.COLOR_RGB2GRAY)

        # predict the emotion
        em_input = tf.reshape(croped_face_img, (1, 48, 48, 1))
        em_output = tf.squeeze(em_classifier(em_input)).numpy()
        output_name = class_names[np.argmax(em_output)]

        cv2.imshow('Croped Face', cv2.flip(croped_face_img, 1))
        mp_drawing.draw_detection(image, detection)

        # flip image to put the text
        image = cv2.flip(image, 1)
        cv2.putText(image, output_name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # break purpositaly to process only one face
        break


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

