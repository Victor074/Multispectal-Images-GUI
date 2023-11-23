from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from tkinter import filedialog
from skimage import io, transform


#define a class to take the image as input and the methods will split the image into different channels
#if another channel is required that operation will be done in the specific method that requieres it
#TODO: add plot option to plot specific wavelenght?
#reference : https://www.neonscience.org/resources/learning-hub/tutorials/plot-spec-sig-tiles-python


class image_process:
    """
    This class contains basic filters
    """
    def __init__(self) -> None:
        self._path = None
        self._img = None

    def start(self, path_to_image, canvas) -> None:
        # if path_to_image is None or path_to_image is not str:
        #     print('Input is not a string')
        #     print(type(path_to_image))
        #     print(path_to_image)
        #     return
        self._path = path_to_image
        self._img = cv2.imread(self._path)
        self.canvas = canvas
        self.b, self.g, self.r = cv2.split(self._img)

    
    def GaussianBlur(self, original: bool = False, resize : bool = True) -> None:
        """
        This filter is used to smooth the image and reduce noise. It is commonly used in multispectral images to remove high-frequency noise.
        Input:
        original : 
        resize :
        """
        global image
        filtered_img = cv2.GaussianBlur(self._img, (5, 5), 0)
        #cv2.imwrite('C:/Users/valc2/Documents/GitHub/Multispectal-Images-GUI/Images/gaussian2.tif',filtered_img)
        if resize :
            filtered_img = cv2.resize(filtered_img, (800, 600))
            self._img = cv2.resize(self._img, (800, 600))
        if original : cv2.imshow('Original Image', self._img)
        cv2.imshow('GaussianBlur Image', filtered_img)
        #self.display_image()
        self._close_windows()

    
    def MedianFilter(self, original: bool = False, resize : bool = True) -> None:
        """
        This filter is used to remove salt-and-pepper noise from an image. It is commonly used in multispectral images to remove impulse noise.

        """
        filtered_img = cv2.medianBlur(self._img, 5)
        if resize :
            filtered_img = cv2.resize(filtered_img, (800, 600))
            self._img = cv2.resize(self._img, (800, 600))
        if original : cv2.imshow('Original Image', self._img)
        
        cv2.imshow('MedianFilter Image', filtered_img)
        self._close_windows()


    def BilateralFilter(self, original: bool = False, resize : bool = True) -> None:
        """
        This filter is used to smooth an image while preserving edges. It is commonly used in multispectral images to reduce noise while preserving important features.
        """
        filtered_img = cv2.bilateralFilter(self._img, 5, 175, 175)
        if resize :
            filtered_img = cv2.resize(filtered_img, (800, 600))
            self._img = cv2.resize(self._img, (800, 600))
        if original : cv2.imshow('Original Image', self._img)
        cv2.imshow('Bilateral Filtered Image', filtered_img)
        self._close_windows()
    

    def get_rgb_channels(self, original: bool = False, resize : bool = True) -> None:
        if resize :
            self._img = cv2.resize(self._img, (800, 600))
            b = cv2.resize(self.b, (800, 600))
            g = cv2.resize(self.g, (800, 600))
            r = cv2.resize(self.r, (800, 600))
        if original : cv2.imshow('Original Image', self._img)
        cv2.imshow('B channel', b)
        cv2.imshow('G channel', g)
        cv2.imshow('R channel', r)
        self._close_windows()
    

    def sobel(self) -> None:
        """
        This filter is used to detect edges in an image. It is commonly used in multispectral images for edge detection.
        """
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        # Apply the Sobel filter
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        filtered_img = cv2.medianBlur(sobelx, 5)
        


        #resize
        sobelx = cv2.resize(sobelx, (800, 600))
        sobely = cv2.resize(sobely, (800, 600))
        # Display the original and filtered images
        cv2.imshow('Original Image', self._img)
        cv2.imshow('Sobel X', sobelx)
        cv2.imshow('Sobel Y', sobely)
        self._close_windows()


    def laplacian_filter(self) -> None:
        """
        This filter is used to detect edges and features in an image. It is commonly used in multispectral images for feature extraction.
        """
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        # Apply the Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.resize(laplacian, (800, 600))
        cv2.imshow('Laplacian filter', laplacian)
        self._close_windows()

    
    def get_wavelenghts(self, lower : int = 400, upper : int = 500) -> None:
        hsv_image = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
        mask = np.logical_and(hsv_image[..., 0] >= lower/2, hsv_image[..., 0] <= upper/2)
        filtered_image = cv2.bitwise_and(self._img, self._img, mask=mask.astype(np.uint8))
        cv2.imshow('Filtered Image', filtered_image)
        self._close_windows()

    
    def plot_waves(self):
        img = cv2.imread(self._path, cv2.IMREAD_UNCHANGED)
    
    def Convolution_2D(self, resize = True):
        # Define a kernel (this is a simple averaging kernel)
        kernel = np.ones((5,5),np.float32)/25

        kernel2 = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])
        # Apply the kernel to the image
        filtered_image = cv2.filter2D(self._img, -1, kernel2)
        if resize :
            filtered_image = cv2.resize(filtered_image, (800, 600))
            #self._img = cv2.resize(self._img, (800, 600))
        cv2.imshow('Filtered Image', filtered_image)
        self._close_windows()

        

    def _close_windows(self) -> None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #---------------------------------------------------------

    # Define the filters
    def natural_color(self):
        # This filter combines the red, green and blue bands to create a natural color image
        global image
        image = Image.open(self._path)
        image = image.resize((512,512))
        image = image.convert("RGB")
        image = image.crop((0, 0, 512, 512)) # Crop the image to fit the canvas
        #return image
        self.display_image()

    def false_color(self):
        # This filter combines the near infrared, red and green bands to create a false color image
        # It highlights vegetation in bright green and water in dark blue
        global image
        image = Image.open(self._path)
        image = image.convert("RGB")
        image = image.resize((512,512))
        r, g, b = image.split()
        nir = b # Use the blue band as near infrared
        image = Image.merge("RGB", (nir, r, g))
        image = image.crop((0, 0, 512, 512)) # Crop the image to fit the canvas
        #return image
        self.display_image()

    def thermal(self):
        # This filter uses the thermal infrared band to create a grayscale image
        # It shows the temperature variations of the surface
        global image
        image = Image.open(self._path)
        image = image.resize((512,512))
        image = image.convert("L") # Convert to grayscale
        t = image.split()[0] # Use the first band as thermal infrared
        image = ImageOps.equalize(t) # Enhance the contrast
        image = image.crop((0, 0, 512, 512)) # Crop the image to fit the canvas
        #return image
        #image = self.resize(image)
        self.display_image()
    
    def display_image(self):
        # This function displays the image on the canvas
        global photo
        photo = ImageTk.PhotoImage(image) # Convert the image to a tkinter object
        self.canvas.create_image(256, 256, image=photo) # Display the image at the center of the canvas
    
    def resize(self, img):
        # Resize the image
        new_shape = (128, 128)
        resized_img = transform.resize(img, new_shape)
        return resized_img



class GUI:
    def __init__(self) -> None:
        self.filters = image_process()

    def start(self):
        # Create the main window
        window = tk.Tk()
        window.title("Multispectral Imaging Filters")
        # Create a canvas to display the image
        self.canvas = tk.Canvas(window, width=512, height=512)
        self.canvas.pack()
        # Create a frame to hold the buttons
        self.frame = tk.Frame(window)
        self.frame.pack()
        # Create the buttons
        
        # Ask the user to select an image file
        self.filename = tk.filedialog.askopenfilename(title="Select an image file")    
        print(self.filename)
        self.filters.start(self.filename, self.canvas)
        self.buttons()
        self.filters.natural_color
        
        window.mainloop()
        #return self.filename
    
    def buttons(self) -> None:
        # Create buttons for each filter
        natural_button = tk.Button(self.frame, text="Natural Color", command=self.filters.natural_color)
        natural_button.pack(side=tk.LEFT)
        
        false_button = tk.Button(self.frame, text="False Color", command=self.filters.false_color)
        false_button.pack(side=tk.LEFT)
        
        thermal_button = tk.Button(self.frame, text="Thermal", command=self.filters.thermal)
        thermal_button.pack(side=tk.LEFT)

        GaussianBlur_button = tk.Button(self.frame, text="GaussianBlur", command=self.filters.GaussianBlur)
        GaussianBlur_button.pack(side=tk.LEFT)

        MedianFilter_button = tk.Button(self.frame, text="MedianFilter", command=self.filters.MedianFilter)
        MedianFilter_button.pack(side=tk.LEFT)

        BilateralFilter_button = tk.Button(self.frame, text="BilateralFilter", command=self.filters.BilateralFilter)
        BilateralFilter_button.pack(side=tk.LEFT)

        get_rgb_channels_button = tk.Button(self.frame, text="get_rgb_channels", command=self.filters.get_rgb_channels)
        get_rgb_channels_button.pack(side=tk.LEFT)

        sobel_button = tk.Button(self.frame, text="sobel", command=self.filters.sobel)
        sobel_button.pack(side=tk.LEFT)
    
        laplacian_filter_button = tk.Button(self.frame, text="laplacian_filter", command=self.filters.laplacian_filter)
        laplacian_filter_button.pack(side=tk.LEFT)

        Convolution_2D_button = tk.Button(self.frame, text="Convolution_2D", command=self.filters.Convolution_2D)
        Convolution_2D_button.pack(side=tk.LEFT)
    




#path = "C:/Users/valc2/Documents/ITESO/TOG/SPOT5.tif"
#path = "C:/Users/valc2/Documents/GitHub/Multispectal-Images-GUI/Images/ri_t_ag1.tif"
#path  = "C:/Users/valc2/Downloads/jojo.jpg"
#path = "C:/Users/valc2/Documents/GitHub/Multispectal-Images-GUI/Images/a-study-in-algae.tif"
#path = "C:/Users/valc2/Documents/ITESO/TOG/IMS1_HYSI_GEO_107_23MAY2009_S6_RADIANCE_02_SPBIN/107_23MAY2009_S6_RADIANCE_02/IMS1_HYSI_GEO_107_23MAY2009_S6_RADIANCE_02_SPBIN.tif"


# obj = image_process(path)
# obj.sobel()
#obj.BilateralFilter()
#obj.GaussianBlur(original=True,resize=True)
obj = GUI()
obj.start()