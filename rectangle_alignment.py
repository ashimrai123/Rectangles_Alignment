import cv2
import numpy as np

# Load the image with rectangles in random positions and angles
input_image = cv2.imread('rectangles.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply edge detection to find contours
edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the maximum width and height of the aligned image
aligned_width = max([cv2.boundingRect(contour)[2] for contour in contours if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 4])
aligned_height = max([cv2.boundingRect(contour)[3] for contour in contours if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 4])

# Set the desired spacing and padding between rectangles
spacing = 20  # Adjust this value as needed
padding = 50  # Adjust this value as needed

# Calculate the number of rows and columns in the gallery
num_rows = len(contours)
num_cols = 2  # You can adjust this based on the desired number of columns

# Calculate the total width and height of the aligned image
total_width = num_cols * aligned_width + (num_cols - 1) * spacing + 2 * padding
total_height = num_rows * aligned_height + (num_rows - 1) * spacing + 2 * padding

# Create a blank image to draw aligned rectangles
aligned_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

x_offset = padding  # Initialize x offset
y_offset = padding  # Initialize y offset

for contour in contours:
    # Approximate the contour to a rectangle
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        # Sort the vertices of the rectangle in clockwise order
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Find the longer side of the rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        if width < height:
            width, height = height, width
            # Rotate the rectangle by 90 degrees
            box = np.roll(box, 1, axis=0)

        src_pts = box.astype("float32")

        # Calculate the destination points for the aligned rectangle
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the perspective transformation to align the rectangle
        aligned_rect = cv2.warpPerspective(input_image, M, (width, height))

        # Place the aligned rectangle onto the aligned image with spacing and padding
        aligned_image[y_offset:y_offset + height, x_offset:x_offset + width] = aligned_rect

        x_offset += aligned_width + spacing  # Update x offset for the next rectangle with spacing

        # Move to the next row if needed
        if x_offset + aligned_width + padding > total_width:
            x_offset = padding
            y_offset += aligned_height + spacing

# Save the aligned rectangles image
cv2.imwrite('aligned_rectangles.png', aligned_image)

# Display the aligned image
cv2.imshow('Aligned Rectangles', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
