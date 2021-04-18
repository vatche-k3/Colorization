from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

class Colorizer():
    CLUSTERS = None    
    def __init__(self, clusters = 5):
        self.CLUSTERS = clusters

    # Converts original image into grayscale image
    def create_grayscale_image(self):
        # Open image
        image = Image.open("images.jpeg")

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
        grayscale_image.save('new.png')
        return grayscale_image

    # Gets the five most representative colors of the image using k means clustering
    def getDominantColors(self):
         # Get the image
        img = cv2.imread("images.jpeg")
        
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

    def getSurroundingGrid(i, j, image):
        # Initialize an array to get all surrounding pixels of given coordinates i, j
        array = np.zeros((3, 3), dtype=list)
        image_rgb = image.convert("RGB")
        # All possible surroinding pixels
        coordinates = [(j-1,i-1),(j,i-1),(j+1,i-1),(j-1,i),(j,i),(j+1,i),(j-1,i+1),(j,i+1),(j+1,i+1)]
        c = 0
        for x in range(3):
            for y in range(3):
                    
                    # Set pixel as black if it does not have 9 surrounding pixels, i.e. a border
                    if(coordinates[c][1] < 0 or coordinates[c][0] >= image.width or coordinates[c][1] >= image.height):
                        # print("here1")
                        array[x][y] = (256,256,256)
                    else:
                        # Get the rgb value of all the surrounding pixels
                        # print("here2")
                        # print(image.size)
                        array[x][y] = image_rgb.getpixel(coordinates[c])
                    c=c+1

        return array
    
    def getSixPatches(self):
        # Open image
        image = Image.open("new.png")
        # Get width and height
        width, height = image.size
        # Get width of second half of the image
        half = (int)(width/2)
        # Initialize array to contain the bw patches of the second half of the image
        data = np.empty((height, width-half), dtype=list)
        # print(len(data), len(data[0]))
        for i in range(height):
            for j in range(half, width):
                # print(i,j)
                # print(i,j-half)
                data[i][j-half] = Colorizer.getSurroundingGrid(i, j, image)
                #print(data[i][j-half])
        return 0
    
    # Function creates a new image out of the five dominant colors
    def create_representative_image(self, colors):
        image = Image.open("images.jpeg")
        image_rgb = image.convert("RGB")
        width, height = image.size
        data = np.zeros((height, width, 3), dtype=np.uint8)

        # Take each rgb value for each pixel and grabs the closest representative color creating a new image and returning
        for i in range(width):
            for j in range(height):
                rgb_pixel_value = image_rgb.getpixel((i,j))
                data[j][i] = self.closestColor(colors, rgb_pixel_value)        
        test_image = Image.fromarray(data)
        test_image.save('representativeImg.png')
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
        
if __name__ == '__main__':
    imageColorizer = Colorizer()
    grayscale_image = imageColorizer.create_grayscale_image()
    # grayscale_image.show()
    domColors = imageColorizer.getDominantColors()
    print(domColors)
    testImage = imageColorizer.create_representative_image(domColors)
    testImage.show()
    sixPatches = imageColorizer.getSixPatches()