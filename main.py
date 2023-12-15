import cv2
from datetime import datetime

# Get current date and time
now = datetime.today()

# Import FaceDetector from cvzone.FaceDetectionModule
from cvzone.FaceDetectionModule import FaceDetector

# Create a FaceDetector instance
detector = FaceDetector()

# Function to draw rectangles around detected faces
def draw(img, classifier, scaleFactor, minNeighbors, color, text, date):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    
    # List to store coordinates of detected faces
    coords = []
    
    # Loop through detected faces and draw rectangles
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        coords = [x, y, w, h]
    
    # Put timestamp on the image
    cv2.putText(img, date, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    return img, coords

# Function to detect faces in the image
def detect(img, face_cascade):
    # Get the current time
    currentTime = datetime.now()
    
    # Format the time as a string
    timestampMessage = currentTime.strftime("%Y/%m/%d   %H: %M : %S")
    
    # Call the draw function to draw rectangles around faces
    img, coords = draw(img, face_cascade, 1.1, 10, (0, 0, 250), "Face", timestampMessage)
    
    return img

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Infinite loop to capture frames and detect faces
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Call the detect function to detect faces in the frame
    frame = detect(frame, detector)
    
    # Display the frame with detected faces
    cv2.imshow('frame', frame)

    # Check for key presses
    key = cv2.waitKey(1)
    
    # If 'q' is pressed, exit the loop
    if key & 0xFF == ord('q'):
        break
    # If 's' is pressed, save the frame as 'face.jpg'
    elif key & 0xFF == ord('s'):
        cv2.imwrite('face.jpg', frame)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
