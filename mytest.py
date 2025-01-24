import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load pre-trained model and face detection classifier
face_detector = cv2.CascadeClassifier('./haarcascade_face_default.xml')
emotion_model = load_model('./Emotion_Detection.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Surprise', 'Sad']

# Access the webcam
camera = cv2.VideoCapture(0)

def preprocess_face(face_region):
    """
    Preprocesses a detected face for emotion prediction.
    """
    face_resized = cv2.resize(face_region, (48, 48), interpolation=cv2.INTER_AREA)
    face_normalized = face_resized.astype('float') / 255.0
    face_array = img_to_array(face_normalized)
    return np.expand_dims(face_array, axis=0)

def display_emotion(frame, x, y, emotion):
    """
    Draws a rectangle around the detected face and displays the predicted emotion.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

while True:
    # Capture a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face_region = gray_frame[y:y + h, x:x + w]

        if np.sum(face_region) != 0:
            face_input = preprocess_face(face_region)
            prediction = emotion_model.predict(face_input)[0]
            emotion_index = np.argmax(prediction)
            emotion_text = emotion_labels[emotion_index]
            display_emotion(frame, x, y, emotion_text)
        else:
            cv2.putText(frame, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
