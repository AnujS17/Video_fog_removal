#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import cv2

# open the video capture object
video_capture = cv2.VideoCapture("D:\DIP Project\hazy_video1.mp4")

# check if the video is opened successfully
if not video_capture.isOpened():
    print("Error opening video file")

# loop through the video frames
while video_capture.isOpened():
    # read the next frame
    ret, frame = video_capture.read()
    if ret:
        # show the frame
        cv2.imshow('Frame', frame)
        # press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        # end the loop if the video is finished
        break

# release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

def estimate_atmospheric_light(img, mean_of_top_percentile=0.1):
    # Find the number of pixels to take from the top percentile
    num_pixels = int(np.prod(img.shape[:2]) * mean_of_top_percentile)

    # Find the maximum pixel value for each channel
    max_channel_vals = np.max(np.max(img, axis=0), axis=0)

    # Sort the channel values in descending order
    sorted_vals = np.argsort(max_channel_vals)[::-1]

    # Take the highest pixel values from each channel
    atmospheric_light = np.zeros((1, 1, 3), np.uint8)
    for channel in range(3):
        atmospheric_light[0, 0, channel] = np.sort(img[:, :, sorted_vals[channel]].ravel())[-num_pixels]

    return atmospheric_light

def fast_visibility_restoration(frame, atmospheric_light, tmin=0.1, A=1.0, omega=0.95, guided_filter_radius=40, gamma=0.7):
    # Normalize the frame and atmospheric light
    normalized_frame = frame.astype(np.float32) / 255.0
    normalized_atmospheric_light = atmospheric_light.astype(np.float32) / 255.0

    # Compute the transmission map
    transmission_map = 1 - omega * cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2GRAY) / cv2.cvtColor(normalized_atmospheric_light, cv2.COLOR_BGR2GRAY)

    # Apply the soft matting guided filter to the transmission map
    guided_filter = cv2.ximgproc.createGuidedFilter(normalized_frame, guided_filter_radius, eps=1.0)
    transmission_map = guided_filter.filter(transmission_map)

    # Apply the gamma correction to the transmission map
    transmission_map = np.power(transmission_map, gamma)

    # Threshold the transmission map to ensure a minimum value
    transmission_map = np.maximum(transmission_map, tmin)

    # Compute the dehazed image
    dehazed_frame = (normalized_frame - normalized_atmospheric_light) / np.expand_dims(transmission_map, axis=2) + normalized_atmospheric_light

    # Apply the A parameter to the dehazed image
    dehazed_frame = A * dehazed_frame

    # Normalize the dehazed image and convert to 8-bit color
    dehazed_frame = np.uint8(np.clip(dehazed_frame * 255.0, 0, 255))

    return dehazed_frame

# Read the video file
hazy_video = cv2.VideoCapture("D:\DIP Project\hazy_video1.mp4")

# Read the first frame and use its maximum pixel values as the atmospheric light
ret, frame = hazy_video.read()
atmospheric_light = estimate_atmospheric_light(frame)

prev_frame = np.zeros((480, 640, 3), np.uint8)

while True:
    # Read the next frame from the video
    ret, frame = hazy_video.read()

    if not ret:
        break

    # Perform fast visibility restoration on the current frame
    dehazed_frame = fast_visibility_restoration(frame, atmospheric_light)

    # Display the dehazed frame
    cv2.imshow('Dehazed Frame', dehazed_frame)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video file and close all windows
hazy_video.release()
cv2.destroyAllWindows()


calculate_video_metrics("D:\DIP Project\hazy_video.mp4", 'dehazed_frame')


# In[ ]:




