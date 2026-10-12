import numpy as np
import cv2
from typing import List, Tuple, Optional

from best_fit import fit_and_visualize_lines

class VideoTracker:
    def __init__(self, video_path: str, pixel_distance_threshold: int = 500, splits: int = 30,
                 left_limit_ratio: float = 0.1, right_limit_ratio: float = 0.1):
        
        self.video_path = video_path
        self.pixel_distance_threshold = pixel_distance_threshold
        self.splits = splits
        self.cap = None
        self.use_gpu = self._check_gpu_support()
        
        # Limit parameters (ratios of image width)
        self.left_limit_ratio = left_limit_ratio  # 0.1 = 10% from left
        self.right_limit_ratio = right_limit_ratio  # 0.1 = 10% from right
        
        # Dynamic limits calculated from bottom pixels
        self.left_limit = None
        self.right_limit = None
        
        # Flag to track if limits have been initialized
        self.limits_initialized = False
        
    def _check_gpu_support(self) -> bool:
        
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

    def _gpu_preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (3, 3), 5)
        edges = cv2.Canny(blurred, 100, 200)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)

        min_size = 10
        filtered_edges = np.zeros_like(edges, dtype=np.uint8)

        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_edges[labels == i] = 255

        return filtered_edges

    def _find_bottom_white_pixels(self, image: np.ndarray, use_limits: bool = True) -> List[Tuple[int, int]]:
        
        height, width = image.shape
        bottom_white_pixels = []
        top_skip = 0
        bottom_skip = 10
        
        # Calculate search range based on limits
        search_start_x = 0
        search_end_x = width
        
        if use_limits and self.left_limit is not None and self.right_limit is not None:
            search_start_x = max(0, int(self.left_limit))
            search_end_x = min(width, int(self.right_limit))
            print(f"Searching between X={search_start_x} and X={search_end_x}")

        for x in range(search_start_x, search_end_x):
            column = image[top_skip:height - bottom_skip, x]
            white_indices = np.where(column == 255)[0]

            if len(white_indices) > 0:
                bottom_pixel_y = white_indices[-1] + top_skip
                bottom_white_pixels.append((x, bottom_pixel_y))

        return bottom_white_pixels
    
    def _calculate_bottom_limits(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Calculate left and right limits based on bottom white pixels in the frame.
        Returns (left_limit, right_limit)
        """
        # Get height and width - frame is a color image with 3 channels
        height, width, _ = frame.shape  # Changed from frame.shape to include channels
        
        # Find bottom pixels in the entire frame first
        edges = self._gpu_preprocess_frame(frame)
        all_bottom_pixels = self._find_bottom_white_pixels(edges, use_limits=False)
        
        if not all_bottom_pixels:
            # If no pixels found, use default ratios
            left_limit = int(width * self.left_limit_ratio)
            right_limit = int(width * (1 - self.right_limit_ratio))
        else:
            # Find the leftmost and rightmost pixels near the bottom
            # Filter pixels near the bottom (last 20% of height)
            bottom_threshold = int(height * 0.8)
            bottom_pixels = [p for p in all_bottom_pixels if p[1] > bottom_threshold]
            
            if bottom_pixels:
                x_coords = [p[0] for p in bottom_pixels]
                left_limit = min(x_coords)
                right_limit = max(x_coords)
                
                # Add some margin based on ratios
                margin = int(width * 0.05)  # 5% margin
                left_limit = max(0, left_limit - margin)
                right_limit = min(width, right_limit + margin)
            else:
                # Fallback to ratio-based limits
                left_limit = int(width * self.left_limit_ratio)
                right_limit = int(width * (1 - self.right_limit_ratio))
        
        return left_limit, right_limit

    @staticmethod
    def _calculate_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
        
        return int(np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2))

    def _filter_pixels(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        
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
        
        if number_splits == -1:
            number_splits = self.splits

        if not coordinates:
            return [[] for _ in range(number_splits)]

        sorted_coords = sorted(coordinates, key=lambda coord: coord[0])
        split_indices = np.linspace(0, len(sorted_coords), number_splits + 1, dtype=int)
        return [sorted_coords[split_indices[i]:split_indices[i + 1]] for i in range(number_splits)]

    def _calculate_centroids(self, coordinate_lists: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        
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
        
        updated_image = image.copy()

        for i in range(len(coordinates) - 1):
            cv2.line(updated_image, coordinates[i], coordinates[i + 1], color=(0, 255, 0), thickness=2)

        return updated_image
    
    def get_dynamic_peak_point(self, points, top_percent=0.2):

        sorted_points = sorted(points, key=lambda p: p[1])
        
        n_top = max(1, int(len(points) * top_percent))
        top_points = sorted_points[:n_top]

        avg_x = sum(p[0] for p in top_points) / len(top_points)
        avg_y = sum(p[1] for p in top_points) / len(top_points)
        
        return (int(avg_x), int(avg_y))
    
    def draw_limits(self, image: np.ndarray) -> np.ndarray:
        """Draw left and right limit lines on the image"""
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

    def process_video(self):
        
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        try:
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                height, width, _ = frame.shape
                height //= 2
                width //= 2
                frame = cv2.resize(frame, (width, height))

                # Initialize limits only once at the beginning
                if not self.limits_initialized:
                    # Use ratio-based limits first
                    self.left_limit = int(width * self.left_limit_ratio)
                    self.right_limit = int(width * (1 - self.right_limit_ratio))
                    print(f"Initial limits from ratios: Left={self.left_limit}, Right={self.right_limit}")
                    
                    # Optionally update based on first frame detection
                    # Uncomment below if you want to auto-detect on first frame
                    # detected_left, detected_right = self._calculate_bottom_limits(frame)
                    # if detected_left > 0 and detected_right > 0:
                    #     self.left_limit = detected_left
                    #     self.right_limit = detected_right
                    #     print(f"Updated to detected limits: Left={self.left_limit}, Right={self.right_limit}")
                    
                    self.limits_initialized = True

                road_path = np.zeros_like(frame)
                edges = self._gpu_preprocess_frame(frame)

                # Find bottom pixels within limits
                bottom_pixels = self._find_bottom_white_pixels(edges, use_limits=True)
                filtered_bottom_pixels = self._filter_pixels(bottom_pixels)
                sections = self._split_coordinates(filtered_bottom_pixels)
                section_centroids = self._calculate_centroids(sections)

                traced_track = self._connect_coordinates(frame, filtered_bottom_pixels)

                for centroid in section_centroids:
                    cv2.circle(traced_track, centroid, 4, (255, 0, 0), -1)


                second_phase = []
                limit_y = 250

                for centroid in section_centroids:
                    cen_x, cen_y = centroid[0], centroid[1]
                    if cen_y <= limit_y:
                        cen_y = limit_y
                    second_phase.append((cen_x, cen_y))

                second_phase_sorted = sorted(second_phase, key=lambda point: point[0])
                second_phase_sorted.insert(0, (0, height))
                second_phase_sorted.append((width, height))


                for i in range(len(second_phase_sorted) - 1):
                    cv2.line(road_path, second_phase_sorted[i], second_phase_sorted[i + 1], color=(255, 0, 0), thickness=1)

                third_phase = self._split_coordinates(second_phase_sorted, 15)
                third_phase_centroid = self._calculate_centroids(third_phase)
                third_phase_centroid.insert(0, (0, height))
                third_phase_centroid.append((width, height))

                for third_centroid in third_phase_centroid:
                    cv2.circle(traced_track, third_centroid, 5, (0, 0, 255), -1)

                for i in range(len(third_phase_centroid) - 1):
                    cv2.line(road_path, third_phase_centroid[i], third_phase_centroid[i + 1], color=(0, 0, 255), thickness=1)

                final_movement = self.get_dynamic_peak_point(second_phase_sorted)
                print("Optimum direction : ", final_movement, "\n")

                cv2.circle(traced_track, final_movement, 10, (100,100,100), -1)
                traced_track2 = fit_and_visualize_lines(third_phase_centroid, width, height)
                
                # Add limit lines to the display
                traced_track = self.draw_limits(traced_track)
                traced_track2 = self.draw_limits(traced_track2)
                edges = self.draw_limits(edges)
                road_path = self.draw_limits(road_path)

                cv2.imshow('Track Frame', traced_track)
                cv2.imshow('Track Frame2', traced_track2)
                cv2.imshow('Edge Detection', edges)
                cv2.imshow('road_path', road_path)
                
                # Display key controls
                control_text = "Q: Quit | L/l: Left limit | R/r: Right limit | U: Update from frame | I: Use ratio limits"
                cv2.putText(traced_track, control_text, (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):  # Move left limit LEFT
                    self.left_limit = max(0, self.left_limit - 5)
                    print(f"Left limit: {self.left_limit}px")
                elif key == ord('L'):  # Shift+L to move left limit RIGHT
                    self.left_limit = min(self.right_limit - 10, self.left_limit + 5)
                    print(f"Left limit: {self.left_limit}px")
                elif key == ord('r'):  # Move right limit RIGHT
                    self.right_limit = min(width, self.right_limit + 5)
                    print(f"Right limit: {self.right_limit}px")
                elif key == ord('R'):  # Shift+R to move right limit LEFT
                    self.right_limit = max(self.left_limit + 10, self.right_limit - 5)
                    print(f"Right limit: {self.right_limit}px")
                elif key == ord('u'):  # Update limits from current frame detection
                    detected_left, detected_right = self._calculate_bottom_limits(frame)
                    if detected_left > 0 and detected_right > 0:
                        self.left_limit = detected_left
                        self.right_limit = detected_right
                        print(f"Limits updated from frame: Left={self.left_limit}, Right={self.right_limit}")
                elif key == ord('i'):  # Reset to ratio-based limits
                    self.left_limit = int(width * self.left_limit_ratio)
                    self.right_limit = int(width * (1 - self.right_limit_ratio))
                    print(f"Reset to ratio limits: Left={self.left_limit}, Right={self.right_limit}")

        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    video_path = 'sample_vid/test3.mp4'  
    
    # These ratios will be used for initial limits
    tracker = VideoTracker(video_path, 
                          left_limit_ratio=0.3,    # 40% from left
                          right_limit_ratio=0.3)   # 30% from right
    
    # Show what ratios mean
    print(f"Left limit ratio: {tracker.left_limit_ratio} ({(tracker.left_limit_ratio*100):.0f}% from left)")
    print(f"Right limit ratio: {tracker.right_limit_ratio} ({(tracker.right_limit_ratio*100):.0f}% from right)")
    
    tracker.process_video()


if __name__ == "__main__":
    main()