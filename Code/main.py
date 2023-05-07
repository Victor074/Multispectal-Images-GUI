import cv2
import matplotlib.pyplot as plt


#define a class to take the image as input and the methods will split the image into different channels
#if another channel is required that operation will be done in the specific method that requieres it

class image_process:
    def __init__(self, path_to_image) -> None:
        # if path_to_image is None or path_to_image is not str:
        #     print('Input is not a string')
        #     return
        self._path = path_to_image
        self._img = cv2.imread(self._path)
        self.b, self.g, self.r = cv2.split(self._img)

    
    def GaussianBlur(self) -> None:
        filtered_img = cv2.GaussianBlur(self._img, (5, 5), 0)

        # Display the filtered image
        cv2.imshow('Filtered Image', filtered_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def MedianFilter(self) -> None:
        filtered_img = cv2.medianBlur(self._img, 5)

        # Display the filtered image
        cv2.imshow('Filtered Image', filtered_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


path = "C:/Users/valc2/Documents/ITESO/TOG/SPOT5.tif"
obj = image_process(path)
obj.GaussianBlur()
