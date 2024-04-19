from PIL import Image

# Open the image
image = Image.open("DATA_jet_car/Car1.png")

# Get the dimensions of the image (width, height)
width, height = image.size

# Calculate the total number of pixels
total_pixels = width * height

print("The image is", width, "pixels wide and", height, "pixels high.")
print("Total pixels in the image:", total_pixels)
