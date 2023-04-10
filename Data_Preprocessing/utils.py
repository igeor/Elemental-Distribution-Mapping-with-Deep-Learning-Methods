import numpy as np 

def median_filter(image, kernel_size):
    """Apply a median filter to an image of shape (h, w, c).
    
    Applies a median filter to each channel of the image
    by sliding a window of size (kernel_size, kernel_size) over the image and
    replacing each pixel with the median value of the pixels in the window.
    
    Args:       
        image (np.ndarray): The image to be filtered.  
        kernel_size (int): The size of the kernel to be used for filtering.
    Returns:    
        np.ndarray: The filtered image.
    """
    
    # Pad the image with zeros
    padded_image = np.pad(image, kernel_size // 2, mode='constant')
    # Extract the dimensions of the image
    h, w, c = image.shape
    # Create a sliding window view of the padded image
    window_shape = (kernel_size, kernel_size, c)
    window_strides = padded_image.strides
    windowed_image = np.lib.stride_tricks.as_strided(padded_image, shape=(h, w, *window_shape), strides=(window_strides[0], window_strides[1], *window_strides))
    # Apply the median filter to each window
    filtered_image = np.median(windowed_image, axis=(2, 3))
    return filtered_image

