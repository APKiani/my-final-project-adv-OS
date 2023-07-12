import cv2
import time
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Load an image for processing
image_path = "C:/Users/koles/Downloads/peakpx (3).jpg"
image = cv2.imread(image_path)

# CPU Image Processing
def cpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using CPU
    # Example: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time
# GPU Image Processing
def gpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using GPU
    # Example: Convert the image to grayscale
    gray_image = tf.image.rgb_to_grayscale(image)

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time


# Run CPU Image Processing
cpu_result, cpu_execution_time = cpu_image_processing(image)
# Run GPU Image Processing
gpu_result, gpu_execution_time = gpu_image_processing(image)

# Display the results and execution times
cv2.imshow("CPU Result", cpu_result)
cv2.waitKey(0)

print("CPU Execution Time:", cpu_execution_time, "seconds")
print("GPU Execution Time:", gpu_execution_time, "seconds")