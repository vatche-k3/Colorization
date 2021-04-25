from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from numpy.linalg import norm
import time
start_time = time.time()

class Colorizer():
    CLUSTERS = None    
    def __init__(self, clusters = 5):
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
        grayscale_image.save('improvedGrayScale.png')
        return grayscale_image

    # Gets the five most representative colors of the image using k means clustering
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
    
    def checkSimilarity(self, leftSidePatch, rightSidePatch, topSix, j, k):
        # print(leftSidePatch)
        # print(rightSidePatch)
        n = 0.0
        # Euclidian distance of left and right patches
        for x, y in zip(rightSidePatch, leftSidePatch):
            for x2, y2 in zip(x, y):
                # print(x2, y2)
                n += norm(np.array(x2) - np.array(y2))
                # print("This is n: ", n)

        

        # Add value to the array
        
        topSix.append((n, (j, k)))
        # print("TopSix before add and delete: ", topSix)
        if len(topSix) > 6:
            topSix.sort(reverse = True)
            del topSix[0]
        # print("TopSix after add: ", topSix)
        # print(topSix)
            

        return topSix


    def getSurroundingGrid(self, i, j, image, check):
        # Initialize an array to get all surrounding pixels of given coordinates i, j
        array = np.zeros((9), dtype=list)
        image_rgb = image.convert("RGB")
        # All possible surroinding pixels
        # coordinates = [(j-1,i-1),(j,i-1),(j+1,i-1),(j-1,i),(j,i),(j+1,i),(j-1,i+1),(j,i+1),(j+1,i+1)]
        coordinates = [(i-1,j-1),(i,j-1),(i+1,j-1),(i-1,j),(i,j),(i+1,j),(i-1,j+1),(i,j+1),(i+1,j+1)]
        c = 0
        for x in range(9):
            # Set pixel as black if it does not have 9 surrounding pixels, i.e. a border
            # print(coordinates[c])
            if(coordinates[c][1] < 0 or coordinates[c][0] >= image.width or coordinates[c][1] >= image.height):
                # print("here1")
                array[x] = 0
            elif check == 0 and coordinates[c][0] < image.width/2:
                array[x] = 0
            elif check == 1 and coordinates[c][0] < 0:
                array[x] = 0
            else:
                # Get the rgb value of all the surrounding pixels
                # print("here2")
                # print(image.size)
                temp = image_rgb.getpixel(coordinates[c])
                array[x] = temp[0]
            c=c+1

        return array

    def predictImageColor(self, topSix, image_rgb, domColors, finalImageData, coor):
        height = 0
        width = 0
        pixelColors = [0,0,0,0,0]
        for i in topSix:
            width = i[1][0]
            height = i[1][1]
            rgb_pixel_value = image_rgb.getpixel((width,height))
            # print("RGB: ", rgb_pixel_value, i)
            for j in range(len(domColors)):
                # print(domColors[j])
                if (np.array(domColors[j]) == np.array(rgb_pixel_value)).all():
                    # print("They are equal")
                    pixelColors[j] += 1
                    break
        
        max = 0
        maxIndex = 0
        for i in range(len(pixelColors)):
            if pixelColors[i] > max:
                max = pixelColors[i]
                maxIndex = i

        # Check for ties
        count = 0
        for i in range(len(pixelColors)):
            if max == pixelColors[i]:
                count += 1
        
        if count > 1:
            rgb_pixel_value = image_rgb.getpixel((width,height))
            finalImageData[coor[1]][coor[0]] = self.closestColor(domColors, rgb_pixel_value)    
        else:
            finalImageData[coor[1]][coor[0]] = domColors[maxIndex]
            # finalImageData[coor[1]][coor[0]] = [0, 0, 0]
        

        
        # print(pixelColors)
        # print(domColors[maxIndex], (coor[1],coor[0]))
        
        return finalImageData
    
    def getVectorX(self, domColors):
        imageGray = Image.open("improvedGrayScale.png")
        image_rgb_gray = imageGray.convert("RGB")

        imageRep = Image.open("representativeImg.png")
        image_rgb_rep = imageRep.convert("RGB")

        width, height = imageGray.size
        # Get width of second half of the image
        half = (int)(width/2)

        leftSide = {}

        # Get each patch on the left side of the image
        for i in range(half):
            for j in range(height):
                patch = self.getSurroundingGrid(i, j, imageGray, 1)
                colorValue = image_rgb_rep.getpixel((i,j))
                for k in range(len(domColors)):
                    # print(domColors[j])
                    if (np.array(domColors[k]) == np.array(colorValue)).all():
                        # print("They are equal")
                        indexValue = k
                        break   
                leftSide[(i,j)] = (list(patch), indexValue) 
        
        print(leftSide)
        print("The end")
        print("--- %s seconds ---" % (time.time() - start_time))
        return 0
    
    # Function creates a new image out of the five dominant colors
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
        test_image.save('improvedRep.png')
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
    # testImage.show()
    inputData = imageColorizer.getVectorX(domColors)