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
        self.weights = self.initializeWeights()
        self.alpha = 0.1

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


    def getSurroundingGrid(self, i, j, image, check):
        # Initialize an array to get all surrounding pixels of given coordinates i, j
        array = np.zeros((9), dtype=list)
        image_rgb = image.convert("RGB")
        # All possible surroinding pixels
        coordinates = [(i-1,j-1),(i,j-1),(i+1,j-1),(i-1,j),(i,j),(i+1,j),(i-1,j+1),(i,j+1),(i+1,j+1)]
        c = 0
        for x in range(9):
            # Set pixel as black if it does not have 9 surrounding pixels, i.e. a border
            if(coordinates[c][1] < 0 or coordinates[c][0] >= image.width or coordinates[c][1] >= image.height):
                array[x] = 0
            elif check == 0 and coordinates[c][0] < image.width/2:
                array[x] = 0
            elif check == 1 and coordinates[c][0] < 0:
                array[x] = 0
            else:
                # Get the rgb value of all the surrounding pixels
                temp = image_rgb.getpixel(coordinates[c])
                array[x] = temp[0]
            c=c+1
        return array
    
    # returns the one hot encoding of a given index
    def oneHotEncoding(self, index):
        return np.array([int(i == index) for i in range(5)])
    
    # Get all the black and white vectors and their one hot encodings
    def getVectorX(self, domColors):
        imageGray = Image.open("improvedGrayScale.png")
        image_rgb_gray = imageGray.convert("RGB")

        imageRep = Image.open("improvedRep.png")
        image_rgb_rep = imageRep.convert("RGB")

        width, height = imageGray.size
        # Get width of second half of the image
        half = (int)(width/2)

        leftSide = {}
        indexValue = 0

        # Get each patch on the left side of the image
        for i in range(half):
            for j in range(height):
                patch = self.getSurroundingGrid(i, j, imageGray, 1)
                colorValue = image_rgb_rep.getpixel((i,j))
                for k in range(len(domColors)):
                    if (np.array(domColors[k]) == np.array(colorValue)).all():
                        indexValue = k
                        encoding = self.oneHotEncoding(indexValue)
                        break   
                leftSide[(i,j)] = (list(patch), list(encoding)) 
        return leftSide
    
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
    
    # Initializes the weights for each weight vector
    def initializeWeights(self):
        weights = np.zeros((5,9), dtype = np.float32)

        # Initialize weights for each weight vector randomly close to 0
        for i in range(5):
            for j in range(9):
                p = np.random.uniform(-0.5,0.5)
                weights[i][j] = p
        return weights

    # Performs the softmax function
    def softmax(self, x):
        logits = np.zeros((1,5))

        # take dot product of each weight vector compared to x and store in logit vector
        logitIndex = 0

        x = [i/255 for i in x]

        for i in self.weights:
            logits[0][logitIndex] = np.dot(x, i)
            logitIndex = logitIndex + 1

        exps = [np.exp(i) for i in logits]
        sum_of_exps = np.sum(exps)
        softmax = [j/sum_of_exps for j in exps]    
        return softmax

    # Grabs the closest representative color to current color and returns
    def closestColor(self,colors,color):
        colors = np.array(colors)
        color = np.array(color)
        distances = np.sqrt(np.sum((colors-color)**2,axis=1))
        index_of_smallest = np.where(distances==np.amin(distances))
        smallest_distance = colors[index_of_smallest]
        temp = smallest_distance[0]
        return temp

    # compute the derivative of the loss function
    def lossFunctionDerivative(self, softMax, encoding):
        array = np.subtract(softMax, encoding) 
        return list(array)

    # compute the loss function
    def lossFunction(self, softMax, encoding):
        loss = 0
        softMaxLog = np.log(softMax)
        for i in range(5):
            loss += ((-1*encoding[i]) * (softMaxLog[0][i]))
        return self.lossFunctionDerivative(softMax, encoding)

    # trains the model
    def training(self, inputData):
        count = 0
        for i in inputData.values():
            if count == 100:
                break
            softMax = self.softmax(i[0])
            count = count + 1
            lossArray = self.lossFunction(softMax, i[1])
            x = [k/255 for k in i[0]]
            
            for j in range(5):
                array = np.zeros(9)
                for l in range(len(x)):
                    array[l] = lossArray[0][j] * x[l] 
                
                array *= self.alpha
                self.weights[j] -= array

    # colors the image
    def colorImprovedAgent(self, domColors):
        
        imageRep = Image.open("improvedRep.png")
        image_rgb_rep = imageRep.convert("RGB")
        finalImageData = np.array(imageRep)
        width, height = imageRep.size
        imageGray = Image.open("improvedGrayScale.png")
        
        # Get width of second half of the image
        half = (int)(width/2)

        for i in range(half, width):
            for j in range(height):
                patch = self.getSurroundingGrid(i, j, imageGray, 0)
                softMax = self.softmax(patch)
                max = 0
                maxIndex = 0
                for k in range(len(softMax[0])):
                    if softMax[0][k] > max:
                        max = softMax[0][k]
                        maxIndex = k
                
                color = domColors[maxIndex]
                finalImageData[j][i] = color
                finalImage = Image.fromarray(finalImageData)
                finalImage.save('improvedImage.png')

        finalImage.save('improvedImage.png')
        finalImage.show()

if __name__ == '__main__':
    imageColorizer = Colorizer()
    grayscale_image = imageColorizer.create_grayscale_image()
    # grayscale_image.show()
    domColors = imageColorizer.getDominantColors()

    testImage = imageColorizer.create_representative_image(domColors)
    # testImage.show()
    inputData = imageColorizer.getVectorX(domColors)

    imageColorizer.training(inputData)
    
    imageColorizer.colorImprovedAgent(domColors)