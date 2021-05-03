from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from numpy.linalg import norm
import time
start_time = time.time()

class Colorizer():
    CLUSTERS = None    
    def __init__(self, clusters = 15):
        self.CLUSTERS = clusters

    # Converts original image into grayscale image
    def create_grayscale_image(self):
        # Open image
        image = Image.open("images2.jpeg")

        # Convert to rgb image
        image_rgb = image.convert("RGB")

        # Get width and height of image
        width, height = image.size
        gray = 0

        # Initialize numpy array
        data = np.zeros((height, width), dtype=np.uint8)

        # Take each rgb value for each pixel and calculate the corresponding grayscale pixel and store in array
        for i in range(width):
            for j in range(height):
                rgb_pixel_value = image_rgb.getpixel((i,j))
                r = rgb_pixel_value[0]
                g = rgb_pixel_value[1]
                b = rgb_pixel_value[2]
                gray = (0.21 * r) + (0.72 * g) + (0.07 * b)
                data[j][i] = gray        
        
        #Create image from calculated grayscale pixels and return
        grayscale_image = Image.fromarray(data)
        grayscale_image.save('bonusGray.png')
        return grayscale_image

    # Gets the best representative colors of the image using k means clustering
    def getDominantColors(self):
         # Get the image
        img = cv2.imread("images2.jpeg")
        
        # Convert to rgb image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # reshapes 3 channel array into total number of pixels * 3 with three columns representing each rgb value
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # cluster the pixels into 5 seperate clusters
        k = KMeans(n_clusters = self.CLUSTERS)
        k.fit(img)

        # cluster gets our dominant colors through the centroids
        self.COLORS = k.cluster_centers_
        
        # Convert colors to int and return
        return self.COLORS.astype(int)

    # Function creates a new image out of the selected number of dominant colors
    def create_representative_image(self, colors):
        image = Image.open("images2.jpeg")
        image_rgb = image.convert("RGB")
        width, height = image.size
        data = np.zeros((height, width, 3), dtype=np.uint8)

        # Take each rgb value for each pixel and grabs the closest representative color creating a new image and returning
        for i in range(width):
            for j in range(height):
                rgb_pixel_value = image_rgb.getpixel((i,j))
                data[j][i] = self.closestColor(colors, rgb_pixel_value)        
        test_image = Image.fromarray(data)
        test_image.save('repBonusImg.png')
        return test_image

    # Grabs the closest representative color to current color and returns
    def closestColor(self,colors,color):
        colors = np.array(colors)
        color = np.array(color)
        distances = np.sqrt(np.sum((colors-color)**2,axis=1))
        index_of_smallest = np.where(distances==np.amin(distances))
        smallest_distance = colors[index_of_smallest]
        temp = smallest_distance[0]
        return temp
    
    # Elbow Method Helper
    def elbowMethod(self):
        image = Image.open("repBonusImg.png")
        image2 = Image.open("images2.jpeg")
        width, height = image.size
        half = (int)(width/2)
        sumPixels = 0 
        for j in range(half):
            for k in range(height):
                a = np.array(image.getpixel((j,k)))
                b = np.array(image2.getpixel((j,k)))
                sumPixels += np.linalg.norm(a - b)
        print(sumPixels)

if __name__ == '__main__':
    imageColorizer = Colorizer()
    grayscale_image = imageColorizer.create_grayscale_image()
    # grayscale_image.show()
    domColors = imageColorizer.getDominantColors()
    testImage = imageColorizer.create_representative_image(domColors)
    imageColorizer.elbowMethod()
    testImage.show()