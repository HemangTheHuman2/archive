import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from visualize import visualize_trajectory

record_path = []

def initialize_tracker(video_path):
    """Initialize video capture and tracking variables"""
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)  # Use 0 for default camera
    
    # Get initial frame dimensions
    ret, first_frame = cap.read()
    if not ret:
        raise Exception("Could not access camera")
        
    height, width = first_frame.shape[:2]
    
    # Initialize tracking path starting at (0,0)
    current_position = np.array([0.0, 0.0])  # Start at origin (0,0)
    path = [(current_position[0], current_position[1])]
    
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([0], [0], 'b-')
    ax.set_xlim(-50, 50)  # Set initial view range
    ax.set_ylim(-50, 50)
    ax.invert_yaxis()  # Invert Y axis to match image coordinates
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Initialize previous frame data (will be set in first call to process_frame)
    prev_gray = None
    prev_corners = None
    
    # Last reported time
    last_report_time = time.time()
    
    return cap, height, width, path, current_position, feature_params, lk_params, fig, ax, line, prev_gray, prev_corners, last_report_time

def crop_frame(frame):
    if frame is not None:
        y, x, c = frame.shape
    else:
        print("AttributeError: 'NoneType' object has no attribute 'shape'.")
        return False
    frame = frame[int((y/5)*2):int((y/5)*3), int((x/5)*2):int((x/5)*3)]
    return frame

def process_frame(cap, height, width, path, current_position, feature_params, lk_params, fig, ax, line, prev_gray, prev_corners, last_report_time):
    """Process a single frame of video"""
    ret, frame = cap.read()
    
    # frame = crop_frame(frame)  # Uncomment if you want to use cropping
    
    if not ret:
        return False, path, current_position, prev_gray, prev_corners, last_report_time
        
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find corners in the frame
    corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    
    if corners is None:
        return True, path, current_position, gray, corners, last_report_time
        
    # Calculate optical flow
    if prev_gray is not None and prev_corners is not None:
        new_corners, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_corners, None, **lk_params
        )
        
        # Filter out points where the flow wasn't found
        good_new = new_corners[status == 1]
        good_old = prev_corners[status == 1]
        
        if len(good_new) > 0 and len(good_old) > 0:
            # Calculate average movement
            movement = np.mean(good_new - good_old, axis=0)
            
            # Scale down movement for better stability (adjust as needed)
            scale_factor = 0.5
            movement *= scale_factor
            
            # Update current position
            # current_position += movement 
            current_position += np.array([-movement[0], movement[1]])  # To invert left-right
            path.append((current_position[0], current_position[1]))
            
            # Update plot
            path_array = np.array(path)
            line.set_xdata(path_array[:, 0])
            line.set_ydata(path_array[:, 1])
            
            # Adjust plot limits if necessary
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            if current_position[0] < x_min + 10:
                ax.set_xlim(x_min - 20, x_max)
            elif current_position[0] > x_max - 10:
                ax.set_xlim(x_min, x_max + 20)
                
            if current_position[1] < y_min + 10:
                ax.set_ylim(y_min - 20, y_max)
            elif current_position[1] > y_max - 10:
                ax.set_ylim(y_min, y_max + 20)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            record_path.append((current_position[0],current_position[1], str(datetime.now().time())))

            # Report current position every second
            current_time = time.time()
            if current_time - last_report_time >= 1.0:  # Report every second
                print(f"Current position: ({current_position[0]:.2f}, {current_position[1]:.2f})")
                last_report_time = current_time
            
            # Draw movement vectors on frame
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
            
            # Show current position on frame
            position_text = f"Position: ({current_position[0]:.2f}, {current_position[1]:.2f})"
            cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
        cv2.imshow('Camera Track', frame)
        
    # Update previous frame and corners
    prev_gray = gray
    prev_corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    
    return True, path, current_position, prev_gray, prev_corners, last_report_time, record_path

def run_tracker():
    """Main function to run the camera tracker"""
    video_path = "test.mp4"
    # Initialize all variables
    cap, height, width, path, current_position, feature_params, lk_params, fig, ax, line, prev_gray, prev_corners, last_report_time = initialize_tracker(video_path)
    
    try:
        while True:
            # Process frame and update variables
            continue_processing, path, current_position, prev_gray, prev_corners, last_report_time, recorded_path = process_frame(
                cap, height, width, path, current_position, 
                feature_params, lk_params, fig, ax, line, prev_gray, prev_corners, last_report_time
            )
            
            if not continue_processing:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Return current position (you can capture this in another script)
            # This is already printed to console but could be returned or saved to file
                
    finally:
        # Print final position
        print(f"Final position: ({current_position[0]:.2f}, {current_position[1]:.2f})")
        cap.release()
        cv2.destroyAllWindows()
        plt.close()
        return current_position, recorded_path  # Return the final position

def get_current_position():
    """Function to get the current camera position"""
    return run_tracker()

def main():
    final_position, saved_path = run_tracker()
    print(f"Tracker finished at position: {final_position}")
    print(f"{saved_path}")
    print("\nVisuals - \n")
    # ani = visualize_trajectory(saved_path)


if __name__ == "__main__":
    main()
