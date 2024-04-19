import torch
import numpy as np
import cv2

# Assuming you have a single RGB image with dimensions 8x8 pixels
height = 8
width = 8
channels = 3  # Three channels for RGB

# Create a random RGB image tensor

image = cv2.imread('BLACKBIRD.png')

#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using OpenCV (optional)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In this example, we're creating a random tensor for demonstration purposes
image_tensor = torch.rand(height, width, channels)

# Flatten the image tensor into a vector
tensor_vector = image_tensor.view(-1)

# Print the shape and content of the tensor vector
print("Shape of tensor vector:", tensor_vector.shape)
print("Content of tensor vector:", tensor_vector)
