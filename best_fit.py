import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

def fit_and_visualize_lines(coordinates, frame_width=800, frame_height=600):

    points = np.array(coordinates)
    
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    mid_point = len(sorted_points) // 2
    first_half = sorted_points[:mid_point]
    second_half = sorted_points[mid_point:]
    
    def fit_line(points):
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        reg = LinearRegression()
        reg.fit(X, y)
        return reg.coef_[0], reg.intercept_
    
    slope1, intercept1 = fit_line(first_half)
    slope2, intercept2 = fit_line(second_half)
    
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    
    def scale_points(points):
        margin = 50
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_scale = (frame_width - 2 * margin) / (x_max - x_min)
        y_scale = (frame_height - 2 * margin) / (y_max - y_min)
        
        scaled_points = np.zeros_like(points)
        scaled_points[:, 0] = (points[:, 0] - x_min) * x_scale + margin
        scaled_points[:, 1] = (points[:, 1] - y_min) * y_scale + margin
        
        return scaled_points, (x_scale, y_scale), (x_min, y_min)
    
    all_points = np.vstack((first_half, second_half))
    scaled_points, (x_scale, y_scale), (x_min, y_min) = scale_points(all_points)
    
    scaled_first_half = scaled_points[:mid_point]
    scaled_second_half = scaled_points[mid_point:]
    
    for point in scaled_first_half:
        cv2.circle(frame, tuple(point.astype(int)), 3, (255, 0, 0), -1)  # Blue
    for point in scaled_second_half:
        cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 0), -1)  # Green
    
    def draw_line(slope, intercept, points, color):
        x_start = points[:, 0].min()
        x_end = points[:, 0].max()
        
        y_start = int(slope * x_start + intercept)
        y_end = int(slope * x_end + intercept)
        
        cv2.line(frame, 
                 (int(x_start), y_start),
                 (int(x_end), y_end),
                 color, 2)

    scaled_slope1 = slope1 * y_scale / x_scale
    scaled_intercept1 = (intercept1 - y_min) * y_scale + 50
    
    scaled_slope2 = slope2 * y_scale / x_scale
    scaled_intercept2 = (intercept2 - y_min) * y_scale + 50
    
    draw_line(scaled_slope1, scaled_intercept1, scaled_first_half, (0, 0, 255))  # Red
    draw_line(scaled_slope2, scaled_intercept2, scaled_second_half, (0, 0, 255))  # Red
    
    return frame

if __name__ == "__main__":
    coordinates = [
        (0, 0), (1, 1), (2, 2), (3, 2.5), (4, 4),  # First half
        (5, 7), (6, 8), (7, 10), (8, 13), (9, 15)  # Second half
    ]
    
    frame = fit_and_visualize_lines(coordinates)
    
    cv2.imshow('Coordinate Analysis', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()