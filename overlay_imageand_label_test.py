import cv2

image_path = 'Highway/images/out-000001.png'  # Specify the path to your image file
txt_file_path = 'Highway/predicted_labels/out-000001.txt'  # Specify the path to your YOLO format labels file



# Load the image
image = cv2.imread(image_path)

# Read the labels from the text file
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Overlay the labels on the image
for line in lines:
    label_data = line.split(" ")
    class_id = int(label_data[0])
    confidence = float(label_data[1])
    x, y, w, h = map(float, label_data[2:])


    # Convert coordinates to integers
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)

    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'Class {class_id} (Confidence: {confidence:.2f})'
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite('image_test.jpg', image)

# Display the image with labels overlay
#cv2.imshow('Image with Labels', image)
#cv2.waitKey(0)
cv2.destroyAllWindows()
