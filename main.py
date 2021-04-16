from PIL import Image
import numpy as np
class Colorizer():
    def __init__(self):
        self   
    def create_image(self):
        image = Image.open("images.jpeg")
        image_rgb = image.convert("RGB")
        width, height = image.size
        gray = 0
        data = np.zeros((height, width), dtype=np.uint8)
        for i in range(width):
            for j in range(height):
                rgb_pixel_value = image_rgb.getpixel((i,j))
                r = rgb_pixel_value[0]
                g = rgb_pixel_value[1]
                b = rgb_pixel_value[2]
                gray = (0.21 * r) + (0.72 * g) + (0.07 * b)
                data[j][i] = gray        
        grayscale_image = Image.fromarray(data)
        grayscale_image.save('new.png')
        return grayscale_image

if __name__ == '__main__':
    imageColorizer = Colorizer()
    grayscale_image = imageColorizer.create_image()
    grayscale_image.show()