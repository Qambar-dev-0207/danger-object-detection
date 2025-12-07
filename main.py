import cv2
import numpy as np
import math

def nothing(x):
    pass

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Window Setup
    cv2.namedWindow("Hand Tracker")
    
    # Create Trackbars for HSV calibration
    # Default values optimized for general skin tone (can be adjusted via UI)
    cv2.createTrackbar("Hue Min", "Hand Tracker", 0, 179, nothing)
    cv2.createTrackbar("Hue Max", "Hand Tracker", 20, 179, nothing)
    cv2.createTrackbar("Sat Min", "Hand Tracker", 50, 255, nothing)
    cv2.createTrackbar("Sat Max", "Hand Tracker", 255, 255, nothing)
    cv2.createTrackbar("Val Min", "Hand Tracker", 50, 255, nothing)
    cv2.createTrackbar("Val Max", "Hand Tracker", 255, 255, nothing)

    # Virtual Object Definition (Center of screen, Radius)
    # Will be set dynamically based on frame size
    obj_center = None
    obj_radius = 40
    
    # Thresholds for states (pixels)
    DANGER_THRESH = 120
    WARNING_THRESH = 250

    while True:
        # 1. FPS Calculation
        timer = cv2.getTickCount()

        # 2. Read Frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        if obj_center is None:
            obj_center = (width // 2, height // 2)

        # 3. Pre-processing & HSV Segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get Trackbar positions
        h_min = cv2.getTrackbarPos("Hue Min", "Hand Tracker")
        h_max = cv2.getTrackbarPos("Hue Max", "Hand Tracker")
        s_min = cv2.getTrackbarPos("Sat Min", "Hand Tracker")
        s_max = cv2.getTrackbarPos("Sat Max", "Hand Tracker")
        v_min = cv2.getTrackbarPos("Val Min", "Hand Tracker")
        v_max = cv2.getTrackbarPos("Val Max", "Hand Tracker")

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Create Mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Morphological operations to remove noise (classical CV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 4. Contour Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_pos = None
        state = "SAFE"
        
        if contours:
            # Find largest contour (assume it's the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Filter small noise
            if cv2.contourArea(max_contour) > 1000:
                # Find Centroid
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hand_pos = (cx, cy)

                    # Draw Contour & Centroid
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 255), 2)
                    cv2.circle(frame, hand_pos, 8, (255, 0, 255), -1)
                    
                    # Draw Line to Object
                    cv2.line(frame, hand_pos, obj_center, (200, 200, 200), 1)

        # 5. Distance Logic & State Determination
        dist = 9999 # Default far
        if hand_pos:
            dist = math.sqrt((hand_pos[0] - obj_center[0])**2 + (hand_pos[1] - obj_center[1])**2)
            
            if dist < DANGER_THRESH:
                state = "DANGER"
            elif dist < WARNING_THRESH:
                state = "WARNING"
            else:
                state = "SAFE"

        # 6. Visual Feedback
        # Set colors based on state
        if state == "SAFE":
            color = (0, 255, 0) # Green
            obj_color = (0, 255, 0)
            border_thickness = 2
        elif state == "WARNING":
            color = (0, 255, 255) # Yellow
            obj_color = (0, 255, 255)
            border_thickness = 4
        else: # DANGER
            color = (0, 0, 255) # Red
            obj_color = (0, 0, 255)
            border_thickness = -1 # Fill object

        # Draw Virtual Object
        cv2.circle(frame, obj_center, obj_radius, obj_color, 2)
        if state == "DANGER":
             cv2.circle(frame, obj_center, obj_radius + 10, (0, 0, 255), 2) # Pulse effect ring

        # Draw State Text
        cv2.putText(frame, f"STATE: {state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if state == "DANGER":
            # Flash "DANGER DANGER"
            cv2.putText(frame, "DANGER DANGER", (width//2 - 200, height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # FPS Display
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {int(fps)}", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show Mask (Optional, helpful for debugging/tuning)
        # Scale mask to small window in corner
        mask_small = cv2.resize(mask, (width//4, height//4))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[height - mask_small.shape[0]:, 0:mask_small.shape[1]] = mask_bgr
        cv2.putText(frame, "Mask View", (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

        cv2.imshow("Hand Tracker", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
