# Import required libraries
import cv2
import pytesseract

# (Optional) Set up Tesseract path if on Windows
# Uncomment and update the path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load and preprocess the image
image_path = 'image-2.jpg'  # Provide the path to your image file
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image for better OCR results
# Use simple thresholding with binary inverse
threshold_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Display the thresholded image for debugging purposes (optional)
cv2.imshow('Threshold Image', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform OCR using Tesseract
custom_config = r'--oem 3 --psm 6'  # Tesseract OCR configurations
text = pytesseract.image_to_string(threshold_image, config=custom_config)

# Print the recognised text
print("Recognised Text:")
print(text)

# (Optional) Draw bounding boxes around detected text
h, w, _ = image.shape
boxes = pytesseract.image_to_boxes(threshold_image)
for box in boxes.splitlines():
    box = box.split()
    x, y, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    # Tesseract uses bottom-left as the origin, so adjust the coordinates
    cv2.rectangle(image, (x, h - y2), (x2, h - y), (0, 255, 0), 2)

# Display the image with bounding boxes (optional)
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()