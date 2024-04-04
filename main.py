import cv2

# Load the image
img = cv2.imread('object.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian filter to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Load the object detection model
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect objects in the image
objects = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected objects
for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the detected objects
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
