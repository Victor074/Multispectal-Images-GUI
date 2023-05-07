import cv2
import matplotlib.pyplot as plt


#define a class to take the image as input and the methods will split the image into different channels
#if another channel is required that operation will be done in the specific method that requieres it

class image_process:
    """
    This class contains basic filters
    """
    def __init__(self, path_to_image) -> None:
        # if path_to_image is None or path_to_image is not str:
        #     print('Input is not a string')
        #     return
        self._path = path_to_image
        self._img = cv2.imread(self._path)
        self.b, self.g, self.r = cv2.split(self._img)

    
    def GaussianBlur(self, original: bool = False, resize : bool = True) -> None:
        """
        This filter is used to smooth the image and reduce noise. It is commonly used in multispectral images to remove high-frequency noise.
        Input:
        original : 
        resize :
        """
        filtered_img = cv2.GaussianBlur(self._img, (5, 5), 0)
        if resize :
            filtered_img = cv2.resize(filtered_img, (800, 600))
            self._img = cv2.resize(self._img, (800, 600))
        if original : cv2.imshow('Original Image', self._img)
        cv2.imshow('GaussianBlur Image', filtered_img)
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
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

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

    def _close_windows(self) -> None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#path = "C:/Users/valc2/Documents/ITESO/TOG/SPOT5.tif"
path = "C:/Users/valc2/Documents/GitHub/Multispectal-Images-GUI/Images/a-study-in-algae.tif"
obj = image_process(path)
obj.MedianFilter(original=True, resize=False)
#$obj.laplacian_filter()
#obj.get_rgb_channels(original=True)
