#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Define a function to perform image dehazing
def defog_frame(frame, prev_frame, atmospheric_light):
    # Convert the input frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the kernel size for dark channel estimation
    kernel_size = 15

    # Calculate the dark channel
    dark_channel = cv2.erode(gray_frame, np.ones((kernel_size, kernel_size), np.uint8))

    # Define an epsilon value
    epsilon = 0.15

    # Calculate the transmission map
    transmission = 1 - epsilon * cv2.cvtColor(dark_channel, cv2.COLOR_GRAY2BGR) / atmospheric_light

    # Initialize the dehazed frame
    dehazed_frame = np.zeros(frame.shape, frame.dtype)

    # Dehaze the frame
    for i in range(3):
        dehazed_frame[:, :, i] = (frame[:, :, i] - atmospheric_light[i]) / np.maximum(transmission[:, :, i], epsilon) + atmospheric_light[i]

    # Calculate the number of pixels for dark channel sampling
    num_pixels = dark_channel.size
    num_samples = int(num_pixels * 0.01)

    # Flatten the dark channel
    dark_channel_flat = dark_channel.flatten()

    # Get the indices of the darkest pixels
    indices = dark_channel_flat.argsort()[-num_samples:]

    # Update the atmospheric light based on the sampled pixels
    atmospheric_light = np.maximum(atmospheric_light, np.max(frame.reshape(num_pixels, 3)[indices], axis=0))

    # Define a weight for blending with the previous frame
    alpha = 0.8

    # Resize the previous frame to match the current frame size
    prev_frame_resized = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0]))

    # Blend the dehazed frame with the previous frame
    dehazed_frame = alpha * dehazed_frame + (1 - alpha) * prev_frame_resized

    return dehazed_frame, atmospheric_light

# Open a hazy video file for processing
hazy_video = cv2.VideoCapture("D:\\DIP Project\\hazy_video.mp4")

# Read the first frame and estimate the atmospheric light
ret, frame = hazy_video.read()
atmospheric_light = np.max(frame, axis=(0, 1))

# Initialize the previous frame
prev_frame = np.zeros((480, 640, 3), np.uint8)

# Process each frame in the hazy video
while True:
    ret, frame = hazy_video.read()
    if not ret:
        break

    # Dehaze the current frame and update the atmospheric light
    dehazed_frame, atmospheric_light = defog_frame(frame, prev_frame, atmospheric_light)
    prev_frame = dehazed_frame

    # Display the dehazed video
    cv2.imshow('Dehazed Video', dehazed_frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the hazy video
hazy_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Define a function to calculate video quality metrics (PSNR and SSIM)
def calculate_video_metrics(video_file_1, video_file_2):
    # Open the input videos
    cap_1 = cv2.VideoCapture(video_file_1)
    cap_2 = cv2.VideoCapture(video_file_2)

    # Check if the videos were opened successfully
    if not cap_1.isOpened() or not cap_2.isOpened():
        print("Error opening videos")
        return None

    # Initialize variables for metrics calculation
    psnr_total = 0
    ssim_total = 0
    num_frames = 0

    # Loop over frames in the videos
    while True:
        # Read a frame from each video
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()

        # Check if frames were read successfully
        if not ret_1 or not ret_2:
            break

        # Convert frames to grayscale
        gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        # Calculate PSNR and SSIM for this frame
        psnr = cv2.PSNR(gray_1, gray_2)
        ssim_score = ssim(gray_1, gray_2, data_range=gray_2.max() - gray_2.min())

        # Accumulate values
        psnr_total += psnr
        ssim_total += ssim_score
        num_frames += 1

    # Release the input videos
    cap_1.release()
    cap_2.release()

    # Calculate average values
    psnr_avg = psnr_total / num_frames
    ssim_avg = ssim_total / num_frames

    return psnr_avg, ssim_avg

# Calculate PSNR and SSIM between two video files
psnr_avg, ssim_avg = calculate_video_metrics("D:\\DIP Project\\hazy_video1.mp4", 'output.mp4')
print(f"Average PSNR: {psnr_avg}")
print(f"Average SSIM: {ssim_avg}")


# In[ ]:




