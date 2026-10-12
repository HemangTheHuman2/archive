import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import yaml
from pathlib import Path
from best_fit import fit_and_visualize_lines

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    'perspective': {
        'src_top_left': [0.42, 0.65],
        'src_top_right': [0.58, 0.65],
        'src_bottom_right': [0.78, 0.92],
        'src_bottom_left': [0.22, 0.92],
    },
    'detection': {
        'white_l_threshold': 160,
        'yellow_h_range': [10, 45],
        'yellow_s_threshold': 50,
        'sobel_thresh': [15, 120],
        'sliding_windows': 9,
        'window_margin': 100,
        'min_pixels': 40,
    },
    'limits': {
        'min_lane_width_px': 200,
        'max_lane_width_px': 1200,
        'margin_ratio': 0.05,
        'update_frequency': 30,
    },
    'smoothing': {
        'ema_alpha': 0.3,
        'buffer_size': 10,
    },
    'visualization': {
        'lane_color': [0, 255, 0],
        'lane_opacity': 0.3,
        'show_curvature': True,
        'show_offset': True,
        'show_limits': True,
    }
}


# ============================================================
# ENHANCED VIDEO TRACKER WITH AUTO-LIMITS
# ============================================================

class EnhancedVideoTracker:
    def __init__(self, video_path: str, pixel_distance_threshold: int = 500, splits: int = 30,
                 config: Optional[Dict] = None):
        
        self.video_path = video_path
        self.pixel_distance_threshold = pixel_distance_threshold
        self.splits = splits
        self.cap = None
        
        # Load configuration
        self.config = config or DEFAULT_CONFIG
        
        # Lane detection state
        self.left_line = None
        self.right_line = None
        self.lane_detected = False
        self.frame_count = 0
        
        # Automatic limits
        self.left_limit = None
        self.right_limit = None
        self.limits_history = deque(maxlen=10)
        
        # Perspective transformer
        self.M = None
        self.Minv = None
        self.img_size = None
        
        # GPU support check
        self.use_gpu = self._check_gpu_support()
        
    def _check_gpu_support(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("GPU acceleration is available!")
                return True
            else:
                print("No GPU found. Falling back to CPU processing.")
                return False
        except Exception as e:
            print(f"Error checking GPU support: {e}")
            return False
    
    def _compute_perspective_transform(self, img_shape: Tuple[int, int]):
        """Compute perspective transform matrices."""
        h, w = img_shape[:2]
        
        if self.img_size == (w, h) and self.M is not None:
            return
            
        self.img_size = (w, h)
        p = self.config['perspective']
        
        # Source points (trapezoid on road)
        src = np.float32([
            [w * p['src_top_left'][0], h * p['src_top_left'][1]],
            [w * p['src_top_right'][0], h * p['src_top_right'][1]],
            [w * p['src_bottom_right'][0], h * p['src_bottom_right'][1]],
            [w * p['src_bottom_left'][0], h * p['src_bottom_left'][1]],
        ])
        
        # Destination points (bird's-eye view)
        offset = 200
        dst = np.float32([
            [offset, 0],
            [w - offset, 0],
            [w - offset, h],
            [offset, h],
        ])
        
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
    
    def _warp_perspective(self, img: np.ndarray) -> np.ndarray:
        """Transform to bird's-eye view."""
        self._compute_perspective_transform(img.shape)
        return cv2.warpPerspective(img, self.M, self.img_size)
    
    def _unwarp_perspective(self, img: np.ndarray) -> np.ndarray:
        """Transform back to original perspective."""
        return cv2.warpPerspective(img, self.Minv, self.img_size)
    
    def _enhanced_preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing with multi-color thresholding.
        Combines white and yellow lane detection.
        """
        det_cfg = self.config['detection']
        
        # Convert to different color spaces
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        b_channel = lab[:, :, 2]  # Yellow/Blue channel
        
        # 1. White lanes (high lightness)
        white_binary = np.zeros_like(l_channel, dtype=np.uint8)
        white_binary[l_channel > det_cfg['white_l_threshold']] = 255
        
        # 2. Yellow lanes (HLS: hue + saturation)
        yellow_h_min, yellow_h_max = det_cfg['yellow_h_range']
        yellow_hls = np.zeros_like(h_channel, dtype=np.uint8)
        yellow_hls[(h_channel >= yellow_h_min) & 
                  (h_channel <= yellow_h_max) & 
                  (s_channel > det_cfg['yellow_s_threshold'])] = 255
        
        # 3. Yellow lanes (LAB: B channel > 155 is yellowish)
        yellow_lab = np.zeros_like(b_channel, dtype=np.uint8)
        yellow_lab[b_channel > 155] = 255
        
        # Combine yellow detections
        yellow_binary = cv2.bitwise_or(yellow_hls, yellow_lab)
        
        # 4. Sobel X gradient (vertical edges)
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        if np.max(abs_sobelx) > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        else:
            scaled_sobel = np.zeros_like(l_channel, dtype=np.uint8)
        
        sx_min, sx_max = det_cfg['sobel_thresh']
        sobel_binary = np.zeros_like(scaled_sobel, dtype=np.uint8)
        sobel_binary[(scaled_sobel >= sx_min) & (scaled_sobel <= sx_max)] = 255
        
        # Combine all channels
        color_binary = cv2.bitwise_or(white_binary, yellow_binary)
        
        # Final combination: Color OR (gradient AND saturation)
        combined = np.zeros_like(color_binary, dtype=np.uint8)
        combined[(color_binary == 255) | ((sobel_binary == 255) & (s_channel > 50))] = 255
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined
    
    def _find_bottom_white_pixels(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Find bottom-most white pixels in each column within limits."""
        height, width = image.shape
        bottom_white_pixels = []
        top_skip = int(height * 0.1)  # Skip top 10%
        bottom_skip = int(height * 0.1)  # Skip bottom 10%
        
        search_start_x = 0
        search_end_x = width
        
        # Apply automatic limits if set
        if self.left_limit is not None and self.right_limit is not None:
            search_start_x = max(0, self.left_limit)
            search_end_x = min(width, self.right_limit)
        
        for x in range(search_start_x, search_end_x):
            column = image[top_skip:height - bottom_skip, x]
            white_indices = np.where(column == 255)[0]
            
            if len(white_indices) > 0:
                bottom_pixel_y = white_indices[-1] + top_skip
                bottom_white_pixels.append((x, bottom_pixel_y))
        
        return bottom_white_pixels
    
    def _auto_detect_limits(self, warped_binary: np.ndarray, frame: np.ndarray) -> Tuple[int, int]:
        """
        Automatically detect left and right limits based on lane detection.
        Uses histogram analysis of bottom part of warped image.
        """
        height, width = warped_binary.shape[:2]
        limits_cfg = self.config['limits']
        
        # Take histogram of bottom third of warped image
        bottom_third = warped_binary[2*height//3:, :]
        histogram = np.sum(bottom_third, axis=0) // 255
        
        # Smooth histogram
        histogram_smooth = np.convolve(histogram, np.ones(5)/5, mode='same')
        
        # Find peaks for left and right lanes
        midpoint = width // 2
        left_half = histogram_smooth[:midpoint]
        right_half = histogram_smooth[midpoint:]
        
        # Find significant peaks (above threshold)
        threshold = np.max(histogram_smooth) * 0.2
        
        left_peaks = np.where(left_half > threshold)[0]
        right_peaks = np.where(right_half > threshold)[0] + midpoint
        
        if len(left_peaks) > 0 and len(right_peaks) > 0:
            # Use outermost peaks for robust detection
            left_limit = max(0, left_peaks[0] - int(width * limits_cfg['margin_ratio']))
            right_limit = min(width, right_peaks[-1] + int(width * limits_cfg['margin_ratio']))
            
            # Validate lane width
            lane_width = right_limit - left_limit
            if (lane_width >= limits_cfg['min_lane_width_px'] and 
                lane_width <= limits_cfg['max_lane_width_px']):
                
                # Update history for smoothing
                self.limits_history.append((left_limit, right_limit))
                
                # Use median of recent history for stability
                if len(self.limits_history) >= 3:
                    left_limits = [l for l, _ in self.limits_history]
                    right_limits = [r for _, r in self.limits_history]
                    left_limit = int(np.median(left_limits))
                    right_limit = int(np.median(right_limits))
                
                return left_limit, right_limit
        
        # Fallback: If no good detection, use frame edges with margins
        margin = int(width * 0.15)
        return margin, width - margin
    
    @staticmethod
    def _calculate_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
        """Calculate Euclidean distance between two points."""
        return int(np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2))
    
    def _filter_pixels(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter out outlier points based on distance threshold."""
        if len(points) <= 2:
            return points
        
        filtered_points = [points[0]]
        
        for i in range(1, len(points) - 1):
            prev_dist = self._calculate_distance(points[i], points[i - 1])
            next_dist = self._calculate_distance(points[i], points[i + 1])
            
            if prev_dist <= self.pixel_distance_threshold and next_dist <= self.pixel_distance_threshold:
                filtered_points.append(points[i])
        
        filtered_points.append(points[-1])
        return filtered_points
    
    def _split_coordinates(self, coordinates: List[Tuple[int, int]], number_splits=-1) -> List[List[Tuple[int, int]]]:
        """Split coordinates into sections."""
        if number_splits == -1:
            number_splits = self.splits
        
        if not coordinates:
            return [[] for _ in range(number_splits)]
        
        sorted_coords = sorted(coordinates, key=lambda coord: coord[0])
        split_indices = np.linspace(0, len(sorted_coords), number_splits + 1, dtype=int)
        return [sorted_coords[split_indices[i]:split_indices[i + 1]] for i in range(number_splits)]
    
    def _calculate_centroids(self, coordinate_lists: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Calculate centroids of coordinate groups."""
        centroids = []
        
        for sublist in coordinate_lists:
            if not sublist:
                centroids.append((0, 0))
            else:
                centroid = (
                    int(sum(coord[0] for coord in sublist) / len(sublist)),
                    int(sum(coord[1] for coord in sublist) / len(sublist))
                )
                centroids.append(centroid)
        
        return centroids
    
    @staticmethod
    def _connect_coordinates(image: np.ndarray, coordinates: List[Tuple[int, int]]) -> np.ndarray:
        """Connect coordinates with lines."""
        updated_image = image.copy()
        
        for i in range(len(coordinates) - 1):
            cv2.line(updated_image, coordinates[i], coordinates[i + 1], color=(0, 255, 0), thickness=2)
        
        return updated_image
    
    def get_dynamic_peak_point(self, points, top_percent=0.2):
        """Find the optimal direction point."""
        if not points:
            return (0, 0)
        
        sorted_points = sorted(points, key=lambda p: p[1])
        
        n_top = max(1, int(len(points) * top_percent))
        top_points = sorted_points[:n_top]
        
        avg_x = sum(p[0] for p in top_points) / len(top_points)
        avg_y = sum(p[1] for p in top_points) / len(top_points)
        
        return (int(avg_x), int(avg_y))
    
    def draw_limits(self, image: np.ndarray) -> np.ndarray:
        """Draw left and right limit lines on the image."""
        if not self.config['visualization']['show_limits']:
            return image
            
        height, width = image.shape[:2]
        result = image.copy()
        
        if self.left_limit is not None:
            cv2.line(result, (self.left_limit, 0), (self.left_limit, height), (0, 255, 255), 2)
            cv2.putText(result, f"Left: {self.left_limit}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.right_limit is not None:
            cv2.line(result, (self.right_limit, 0), (self.right_limit, height), (0, 255, 255), 2)
            cv2.putText(result, f"Right: {self.right_limit}", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result
    
    def _draw_lane_overlay(self, image: np.ndarray, left_fitx: np.ndarray, 
                          right_fitx: np.ndarray, ploty: np.ndarray) -> np.ndarray:
        """Draw filled lane overlay on image."""
        vis_cfg = self.config['visualization']
        
        # Create blank image for drawing
        color_warp = np.zeros_like(image, dtype=np.uint8)
        
        # Create polygon points
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Fill polygon
        lane_color = tuple(vis_cfg['lane_color'])
        cv2.fillPoly(color_warp, np.int_([pts]), lane_color)
        
        # Combine with original
        opacity = vis_cfg['lane_opacity']
        result = cv2.addWeighted(image, 1, color_warp, opacity, 0)
        
        return result
    
    def process_video(self):
        """Main video processing loop with automatic limit detection."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")
        
        try:
            self.frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Resize frame for faster processing
                height, width, _ = frame.shape
                scale_factor = 0.5
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Enhanced preprocessing with multi-color detection
                enhanced_edges = self._enhanced_preprocess_frame(frame)
                
                # Warp to bird's-eye view for better lane analysis
                warped_binary = self._warp_perspective(enhanced_edges)
                
                # Automatically detect limits (update every N frames)
                if (self.frame_count % self.config['limits']['update_frequency'] == 1 or 
                    self.left_limit is None):
                    
                    left_limit, right_limit = self._auto_detect_limits(warped_binary, frame)
                    self.left_limit = left_limit
                    self.right_limit = right_limit
                    print(f"Frame {self.frame_count}: Auto-detected limits - Left={left_limit}, Right={right_limit}")
                
                # Find bottom white pixels within auto-detected limits
                bottom_pixels = self._find_bottom_white_pixels(enhanced_edges)
                
                if bottom_pixels:
                    # Filter and process the detected points
                    filtered_pixels = self._filter_pixels(bottom_pixels)
                    sections = self._split_coordinates(filtered_pixels)
                    section_centroids = self._calculate_centroids(sections)
                    
                    # Create traced track visualization
                    traced_track = self._connect_coordinates(frame, filtered_pixels)
                    
                    # Draw centroids
                    for centroid in section_centroids:
                        cv2.circle(traced_track, centroid, 4, (255, 0, 0), -1)
                    
                    # Process for road path visualization
                    second_phase = []
                    limit_y = 250
                    
                    for centroid in section_centroids:
                        cen_x, cen_y = centroid[0], centroid[1]
                        if cen_y <= limit_y:
                            cen_y = limit_y
                        second_phase.append((cen_x, cen_y))
                    
                    # Sort and add boundary points
                    second_phase_sorted = sorted(second_phase, key=lambda point: point[0])
                    second_phase_sorted.insert(0, (0, new_height))
                    second_phase_sorted.append((new_width, new_height))
                    
                    # Create road path visualization
                    road_path = np.zeros_like(frame)
                    for i in range(len(second_phase_sorted) - 1):
                        cv2.line(road_path, second_phase_sorted[i], second_phase_sorted[i + 1], 
                                color=(255, 0, 0), thickness=1)
                    
                    # Further processing for smoother curves
                    third_phase = self._split_coordinates(second_phase_sorted, 15)
                    third_phase_centroid = self._calculate_centroids(third_phase)
                    third_phase_centroid.insert(0, (0, new_height))
                    third_phase_centroid.append((new_width, new_height))
                    
                    # Draw third phase centroids
                    for third_centroid in third_phase_centroid:
                        cv2.circle(traced_track, third_centroid, 5, (0, 0, 255), -1)
                    
                    # Draw third phase lines
                    for i in range(len(third_phase_centroid) - 1):
                        cv2.line(road_path, third_phase_centroid[i], third_phase_centroid[i + 1], 
                                color=(0, 0, 255), thickness=1)
                    
                    # Find optimal direction point
                    final_movement = self.get_dynamic_peak_point(second_phase_sorted)
                    print(f"Frame {self.frame_count}: Optimum direction: {final_movement}")
                    
                    # Mark optimal direction
                    cv2.circle(traced_track, final_movement, 10, (100, 100, 100), -1)
                    
                    # Use best_fit for polynomial fitting
                    traced_track2 = fit_and_visualize_lines(third_phase_centroid, new_width, new_height)
                    
                    # Draw automatic limits
                    traced_track = self.draw_limits(traced_track)
                    traced_track2 = self.draw_limits(traced_track2)
                    enhanced_edges = self.draw_limits(enhanced_edges)
                    road_path = self.draw_limits(road_path)
                    
                    # Display results
                    cv2.imshow('Enhanced Lane Detection', traced_track)
                    cv2.imshow('Polynomial Fit', traced_track2)
                    cv2.imshow('Enhanced Edges', enhanced_edges)
                    cv2.imshow('Road Path', road_path)
                    
                    # Add status information
                    status_text = f"Frame: {self.frame_count} | Limits: Auto | Points: {len(bottom_pixels)}"
                    cv2.putText(traced_track, status_text, (10, new_height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # No points detected
                    print(f"Frame {self.frame_count}: No lane points detected")
                    cv2.imshow('Enhanced Lane Detection', frame)
                
                # Control handling
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Manual override mode
                    print("Manual override - press L/R to adjust, A to return to auto")
                    self._manual_limit_adjustment(frame)
                elif key == ord('s'):  # Save current frame
                    cv2.imwrite(f"frame_{self.frame_count}.jpg", frame)
                    print(f"Frame {self.frame_count} saved")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def _manual_limit_adjustment(self, frame: np.ndarray):
        """Temporary manual adjustment mode."""
        height, width = frame.shape[:2]
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('l'):  # Move left limit left
                self.left_limit = max(0, self.left_limit - 10)
                print(f"Left limit: {self.left_limit}")
            elif key == ord('L'):  # Move left limit right
                self.left_limit = min(self.right_limit - 50, self.left_limit + 10)
                print(f"Left limit: {self.left_limit}")
            elif key == ord('r'):  # Move right limit right
                self.right_limit = min(width, self.right_limit + 10)
                print(f"Right limit: {self.right_limit}")
            elif key == ord('R'):  # Move right limit left
                self.right_limit = max(self.left_limit + 50, self.right_limit - 10)
                print(f"Right limit: {self.right_limit}")
            elif key == ord('a'):  # Return to auto mode
                print("Returning to automatic mode")
                break
            elif key == ord('q'):  # Quit entirely
                return True  # Signal to quit


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main function to run the enhanced lane detection system."""
    
    # Optional: Load custom configuration
    config_path = None  # Set to path if you have a config.yaml
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = DEFAULT_CONFIG
        print("Using default configuration")
    
    # Video path
    video_path = 'sample_vid/sample.mp4'  # Change this to your video path
    
    # Create enhanced tracker
    tracker = EnhancedVideoTracker(
        video_path,
        pixel_distance_threshold=500,
        splits=30,
        config=config
    )
    
    print("\n" + "="*60)
    print("ENHANCED LANE DETECTION SYSTEM")
    print("="*60)
    print("Features:")
    print("- Multi-color thresholding (white & yellow lanes)")
    print("- Automatic left/right limit detection")
    print("- Bird's-eye perspective transform")
    print("- Polynomial curve fitting")
    print("- Temporal smoothing")
    print("\nControls:")
    print("  Q: Quit")
    print("  M: Manual limit adjustment mode")
    print("  S: Save current frame")
    print("="*60 + "\n")
    
    # Process video
    try:
        tracker.process_video()
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()