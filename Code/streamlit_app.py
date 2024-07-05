import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

class ImageProcess:
    def __init__(self):
        self._img = None

    def load_image(self, image_data):
        self._img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        self.b, self.g, self.r = cv2.split(self._img)

    def set_image(self, image):
        self._img = image
        self.b, self.g, self.r = cv2.split(self._img)

    def GaussianBlur(self, ksize=5):
        return cv2.GaussianBlur(self._img, (ksize, ksize), 0)

    def MedianFilter(self, ksize=5):
        return cv2.medianBlur(self._img, ksize)

    def BilateralFilter(self, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(self._img, d, sigma_color, sigma_space)

    def get_rgb_channels(self):
        return cv2.merge([self.r, self.g, self.b])

    def sobel(self):
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.magnitude(sobelx, sobely)

    def histogram_equalization(self):
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def pca(self, n_components=3):
        reshaped_img = self._img.reshape(-1, self._img.shape[2])
        mean_centered = reshaped_img - np.mean(reshaped_img, axis=0)
        cov_matrix = np.cov(mean_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]
        transformed_data = np.dot(mean_centered, eigenvectors)
        transformed_data = (transformed_data - np.min(transformed_data)) / (np.max(transformed_data) - np.min(transformed_data))
        transformed_data = (transformed_data * 255).astype(np.uint8)
        return transformed_data.reshape(self._img.shape[0], self._img.shape[1], n_components)

    def false_color_composite(self):
        return cv2.merge([self.b, self.g, self.r])
    
    def salt_and_pepper_noise(self, amount=0.05, salt_vs_pepper=0.5):
        noisy_img = self._img.copy()
        num_salt = np.ceil(amount * noisy_img.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * noisy_img.size * (1.0 - salt_vs_pepper))

        # Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy_img.shape]
        noisy_img[coords[0], coords[1], :] = 1

        # Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_img.shape]
        noisy_img[coords[0], coords[1], :] = 0

        return noisy_img

    def ndvi(self):
        nir = self.r.astype(float)
        red = self.g.astype(float)
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = ((ndvi + 1) / 2 * 255).astype(np.uint8)
        return ndvi

    def band_ratio(self, band1, band2):
        ratio = band1.astype(float) / band2.astype(float)
        ratio = np.clip(ratio, 0, 1)
        ratio = (ratio * 255).astype(np.uint8)
        return ratio

    def dvi(self):
        nir = self.r.astype(float)
        red = self.g.astype(float)
        dvi = nir - red
        dvi = np.clip(dvi, -255, 255)
        dvi = ((dvi + 255) / 2).astype(np.uint8)
        return dvi

    def evi(self):
        nir = self.r.astype(float)
        red = self.g.astype(float)
        blue = self.b.astype(float)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        evi = np.clip(evi, -1, 1)
        evi = ((evi + 1) / 2 * 255).astype(np.uint8)
        return evi

    def rendvi(self):
        red_edge = self.g.astype(float)
        nir = self.r.astype(float)
        rendvi = (nir - red_edge) / (nir + red_edge)
        rendvi = np.clip(rendvi, -1, 1)
        rendvi = ((rendvi + 1) / 2 * 255).astype(np.uint8)
        return rendvi

# Streamlit UI
st.title("Multispectral Image Processing")
st.markdown("""
This application allows you to apply various filters and processing techniques to multispectral images. It is designed to help visualize and analyze different aspects of the images using OpenCV functions. 
You can upload an image and choose from a range of filters such as Gaussian Blur, Median Filter, Bilateral Filter, Sobel Edge Detection, and more. Each filter is applied using OpenCV functions, and you can save the modified image after applying the desired filters.
""")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

def display_images(original, processed, caption):
    col1, col2 = st.columns(2)
    col1.image(original, caption='Original Image', use_column_width=True)
    col2.image(processed, caption=caption, use_column_width=True)

if uploaded_file is not None:
    img_processor = ImageProcess()
    saved_image = ImageProcess()
    image_data = uploaded_file.read()
    img_processor.load_image(image_data)
    saved_image.load_image(image_data)
    original_img = cv2.cvtColor(img_processor._img, cv2.COLOR_BGR2RGB)
    st.image(original_img, caption='Original Image', use_column_width=True)

    st.sidebar.title("Filters")
    
    if st.sidebar.button('Add Salt and Pepper Noise'):
        noisy_img = img_processor.salt_and_pepper_noise()
        noisy_img_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        st.title("Salt and Pepper Noise")
        display_images(original_img, noisy_img_rgb, 'Salt and Pepper Noise')
        st.write("Salt and Pepper Noise adds random white and black pixels to the image.")

    if st.sidebar.button('Gaussian Blur'):
        processed_img = img_processor.GaussianBlur()
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        saved_image.set_image(saved_image.GaussianBlur())
        st.title("Gaussian Blur")
        display_images(original_img, processed_img_rgb, 'Gaussian Blur')
        st.write("Gaussian Blur is used to smooth the image and reduce noise by applying a Gaussian filter. [OpenCV Documentation](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)")

    if st.sidebar.button('Median Filter'):
        processed_img = img_processor.MedianFilter()
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        saved_image.set_image(saved_image.MedianFilter())
        st.title("Median Filter")
        display_images(original_img, processed_img_rgb, 'Median Filter')
        st.write("Median Filter is used to reduce noise in an image by replacing each pixel's value with the median value of its neighbors. [OpenCV Documentation](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)")

    if st.sidebar.button('Bilateral Filter'):
        processed_img = img_processor.BilateralFilter()
        saved_image.set_image(saved_image.BilateralFilter())
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.title("Bilateral Filter")
        display_images(original_img, processed_img_rgb, 'Bilateral Filter')
        st.write("Bilateral Filter is used to reduce noise while keeping edges sharp by considering both spatial closeness and pixel intensity difference. [OpenCV Documentation](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)")

    if st.sidebar.button('Get RGB Channels'):
        processed_img = img_processor.get_rgb_channels()
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.title("Get RGB Channels")
        display_images(original_img, processed_img_rgb, 'RGB Channels')
        st.write("This operation splits the image into its red, green, and blue channels and then recombines them.")

    if st.sidebar.button('Sobel Filter'):
        processed_img = img_processor.sobel()
        processed_img_rgb = cv2.cvtColor(processed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        st.title("Sobel Filter")
        display_images(original_img, processed_img_rgb, 'Sobel Filter')
        st.write("Sobel Filter is used for edge detection by calculating the gradient of the image intensity at each pixel. [OpenCV Documentation](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)")

    if st.sidebar.button('Histogram Equalization'):
        processed_img = img_processor.histogram_equalization()
        st.title("Histogram Equalization")
        display_images(original_img, processed_img, 'Histogram Equalization')
        st.write("Histogram Equalization enhances the contrast of the image by spreading out the most frequent intensity values. [OpenCV Documentation](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)")

    if st.sidebar.button('PCA'):
        processed_img = img_processor.pca()
        st.title("PCA")
        display_images(original_img, processed_img, 'PCA')
        st.write("Principal Component Analysis (PCA) reduces the dimensionality of the multispectral image while preserving important information.")

    if st.sidebar.button('False Color Composite'):
        processed_img = img_processor.false_color_composite()
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.title("False Color Composite")
        display_images(original_img, processed_img_rgb, 'False Color Composite')
        st.write("False Color Composite creates a false-color image to highlight certain features.")

    if st.sidebar.button('NDVI'):
        ndvi_img = img_processor.ndvi()
        st.title("NDVI")
        display_images(original_img, ndvi_img, 'NDVI')
        st.write("NDVI (Normalized Difference Vegetation Index) is used to analyze remote sensing measurements and assess whether the target being observed contains live green vegetation or not.")
    
    if st.sidebar.button('Band Ratio (Red/Green)'):
        ratio_img = img_processor.band_ratio(img_processor.r, img_processor.g)
        st.title("Band Ratio (Red/Green)")
        display_images(original_img, ratio_img, 'Band Ratio (Red/Green)')
        st.write("Band Ratio (Red/Green) highlights specific features by dividing the values of the red band by the green band.")
    
    if st.sidebar.button('DVI'):
        dvi_img = img_processor.dvi()
        st.title("DVI")
        display_images(original_img, dvi_img, 'DVI')
        st.write("DVI (Difference Vegetation Index) highlights vegetation by subtracting the red band from the near-infrared band.")
    
    if st.sidebar.button('EVI'):
        evi_img = img_processor.evi()
        st.title("EVI")
        display_images(original_img, evi_img, 'EVI')
        st.write("EVI (Enhanced Vegetation Index) reduces noise from atmospheric conditions and soil background signals, providing a clearer picture of vegetation.")
    
    if st.sidebar.button('RENDVI'):
        rendvi_img = img_processor.rendvi()
        st.title("RENDVI")
        display_images(original_img, rendvi_img, 'RENDVI')
        st.write("RENDVI (Red Edge Normalized Difference Vegetation Index) uses the red edge band for better vegetation monitoring.")
    
    if st.button("Save Image"):
        img_rgb = cv2.cvtColor(saved_image._img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Image", data=byte_im, file_name="modified_image.png", mime="image/png")
st.markdown("""
---
For more information and to access the source code, visit the [GitHub repository](https://github.com/Victor074/Multispectal-Images-GUI).
""")