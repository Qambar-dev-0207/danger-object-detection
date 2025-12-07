# Danger Object Detection & Hand Tracker

A real-time computer vision application that detects hand proximity to a virtual object using OpenCV. The system tracks your hand movement and classifies the distance to a central "danger zone" into three states: **SAFE**, **WARNING**, and **DANGER**.

## Features

*   **Real-time Hand Tracking:** Uses HSV color thresholding to segment and track the hand.
*   **Proximity Detection:** Calculates the distance between the hand's centroid and a virtual object.
*   **Dynamic States:**
    *   🟢 **SAFE:** Hand is far from the object.
    *   🟡 **WARNING:** Hand is approaching the object.
    *   🔴 **DANGER:** Hand is too close (visual alarm triggered).
*   **Interactive Calibration:** Built-in trackbars to adjust HSV values for different skin tones and lighting conditions.
*   **Visual Debugging:** Includes a picture-in-picture view of the segmentation mask.

## Prerequisites

*   Python 3.x
*   Webcam

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Qambar-dev-0207/danger-object-detection.git
    cd danger-object-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy
    ```

## Usage

1.  Run the main script:
    ```bash
    python main.py
    ```

2.  **Calibrate:**
    *   A window named "Hand Tracker" will open.
    *   Use the trackbars (Hue Min/Max, Sat Min/Max, Val Min/Max) to adjust the color filter until your hand is clearly white in the bottom-left "Mask View" and the background is black.

3.  **Interact:**
    *   Move your hand towards the circle in the center of the screen.
    *   Observe the state change from Green (Safe) to Yellow (Warning) to Red (Danger).

## Controls

*   Press **`q`** to quit the application.

## How it Works

1.  **Image Capture:** Captures video frames from the webcam.
2.  **Preprocessing:** Converts the frame to HSV color space.
3.  **Segmentation:** Applies a mask based on the calibrated HSV values to isolate the hand.
4.  **Contour Detection:** Finds the largest contour in the mask (assumed to be the hand) and calculates its centroid.
5.  **Logic:** Computes the Euclidean distance between the hand centroid and the screen center.
6.  **Feedback:** Renders the virtual object and status text with appropriate colors based on the distance thresholds.
