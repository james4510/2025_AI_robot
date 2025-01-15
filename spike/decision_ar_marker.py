import time
import cv2 
from picamera2 import Picamera2
import numpy as np
import serial as sr
import marker

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

# Initialize serial communication
ser = sr.Serial('/dev/ttyS0', baudrate=9600, timeout=1)

def make_black(image, threshold=150):    
    """Convert the image to grayscale and create a binary image based on the threshold."""    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    black_image = cv2.inRange(gray_image, threshold, 255)    
    return black_image, gray_image

def path_decision(image, limit=150, force_forward=False):    
    """Analyze the bottom portion of the image to decide the direction."""    
    height, width = image.shape    
    image = image[height-limit:height-10, :]    
    height = limit - 1    
    width = width - 1    
    image = np.flipud(image)        
    mask = image != 0    
    white_distance = np.where(mask.any(axis=0), mask.argmax(axis=0), height)    

    # Define regions for left, front, and right with sub-regions
    left_1 = (0, int(width / 6))
    left_2 = (int(width / 6), int(width / 3))
    left_3 = (int(width / 3), int(width / 2))

    front_1 = (int(width / 2), int(width / 2 + width / 6))
    front_2 = (int(width / 2 + width / 6), int(width / 2 + width / 3))
    front_3 = (int(width / 2 + width / 3), int(width / 2 + width / 2))

    right_1 = (int(width / 2 + width / 2), int(width / 2 + width / 2 + width / 6))
    right_2 = (int(width / 2 + width / 2 + width / 6), int(width / 2 + width / 2 + width / 3))
    right_3 = (int(width / 2 + width / 2 + width / 3), width)

    # Calculate sums for each region
    left_1_sum = np.sum(white_distance[left_1[0]:left_1[1]])
    left_2_sum = np.sum(white_distance[left_2[0]:left_2[1]])
    left_3_sum = np.sum(white_distance[left_3[0]:left_3[1]])

    front_1_sum = np.sum(white_distance[front_1[0]:front_1[1]])
    front_2_sum = np.sum(white_distance[front_2[0]:front_2[1]])
    front_3_sum = np.sum(white_distance[front_3[0]:front_3[1]])

    right_1_sum = np.sum(white_distance[right_1[0]:right_1[1]])
    right_2_sum = np.sum(white_distance[right_2[0]:right_2[1]])
    right_3_sum = np.sum(white_distance[right_3[0]:right_3[1]])

    print("Left:", left_1_sum, left_2_sum, left_3_sum)
    print("Front:", front_1_sum, front_2_sum, front_3_sum)
    print("Right:", right_1_sum, right_2_sum, right_3_sum)

    # Decision logic based on sums
    if force_forward:  # Force forward movement if marker 1156 is detected
        decision = 'f'
    else:
        total_left_sum = left_1_sum + left_2_sum + left_3_sum
        total_front_sum = front_1_sum + front_2_sum + front_3_sum
        total_right_sum = right_1_sum + right_2_sum + right_3_sum

        if total_front_sum > 8000:        
            decision = 'f'  # Move forward    
        elif total_left_sum > total_right_sum:        
            decision = 'l'  # Turn left    
        elif total_left_sum <= total_right_sum:        
            decision = 'r'  # Turn right    
        else:        
            decision = 's'  # Stop    

    return decision

def serial_send(key):    
    """Send a key command to the robot via serial communication."""    
    msg = key.encode()    
    ser.write(msg)    
    time.sleep(0.1)

try:    
    while True:        # Capture a frame from the camera        
        frame = picam2.capture_array()        # Convert RGB to BGR and flip the frame        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)        
        frame_bgr_flipped = cv2.flip(frame_bgr, -1)  # Flip the frame        # Process the image to identify the path        
        black_image, gray_image = make_black(frame_bgr_flipped)        

        # Detect markers
        marker_number = marker.marker_detect(black_image)
        force_forward = marker_number == 1156

        decision = path_decision(black_image, force_forward=force_forward)        
        print(f"Decision: {decision}, Marker: {marker_number}")        

        # Send the decision to the robot        
        serial_send(decision)        

        # Display the video feed and processed images        
        cv2.imshow("Frame (Color)", frame_bgr_flipped)        
        cv2.imshow("Grayscale", gray_image)        
        cv2.imshow("Black Image", black_image)        

        # Exit on 'q' key press        
        if cv2.waitKey(1) & 0xFF == ord('q'):            
            break
finally:    
    picam2.stop()    
    cv2.destroyAllWindows()    
    print("Program terminated.")
