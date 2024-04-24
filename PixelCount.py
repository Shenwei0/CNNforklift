from PIL import Image

# Open the image
image = Image.open("/home/raptor/SeasonyCNN/DATA_fruits/train/Tomato 3/r_0_100_jpg.rf.47d92bd94c05d5c090e480311ee6664a.jpg")

# Get the dimensions of the image (width, height)
width, height = image.size

# Calculate the total number of pixels
total_pixels = width * height

print("The image is", width, "pixels wide and", height, "pixels high.")
print("Total pixels in the image:", total_pixels)
