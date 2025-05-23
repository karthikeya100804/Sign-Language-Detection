import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\kommu\\OneDrive\\Desktop\\TARP\\asl_model.keras", compile=False)
model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])

# Class labels
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess frame
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Get predictions
    preds = model.predict(img, verbose=0)
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)

    # Draw predictions on frame
    label = f"{class_names[predicted_class]} ({confidence*100:.2f}%)"
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
