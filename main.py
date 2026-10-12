import numpy as np
import cv2
from typing import List, Tuple
from best_fit import fit_and_visualize_lines

class VideoTracker:
    def __init__(self, video_path: str, pixel_distance_threshold: int = 500, splits: int = 30):
        self.video_path = cv2.VideoCapture(0)
        self.pixel_distance_threshold = pixel_distance_threshold
        self.splits = splits
        self.cap = None
        self.use_gpu = self._check_gpu_support()
        
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
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_edges[labels == i] = 255
        return filtered_edges

    def _find_bottom_white_pixels(self, image: np.ndarray) -> List[Tuple[int, int]]:
        height, width = image.shape
        bottom_white_pixels = []
        top_skip = 0
        bottom_skip = 10
        for x in range(width):
            column = image[top_skip:height-bottom_skip, x]
            white_indices = np.where(column == 255)[0]
            if len(white_indices) > 0:
                bottom_pixel_y = white_indices[-1] + top_skip
                bottom_white_pixels.append((x, bottom_pixel_y))
        return bottom_white_pixels

    @staticmethod
    def _calculate_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
        return int(np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2))

    def _filter_pixels(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(points) <= 2:
            return points
        filtered_points = [points[0]]
        for i in range(1, len(points) - 1):
            prev_dist = self._calculate_distance(points[i], points[i-1])
            next_dist = self._calculate_distance(points[i], points[i+1])
            if prev_dist <= self.pixel_distance_threshold and next_dist <= self.pixel_distance_threshold:
                filtered_points.append(points[i])
        filtered_points.append(points[-1])
        return filtered_points

    def _split_coordinates(self, coordinates: List[Tuple[int, int]], number_splits=-1) -> List[List[Tuple[int, int]]]:
        if number_splits==-1:
            number_splits = self.splits
        if not coordinates:
            return [[] for _ in range(number_splits)]
        sorted_coords = sorted(coordinates, key=lambda coord: coord[0])
        split_indices = np.linspace(0, len(sorted_coords), number_splits + 1, dtype=int)
        return [sorted_coords[split_indices[i]:split_indices[i+1]] for i in range(number_splits)]

    def _calculate_centroids(self, coordinate_lists: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        centroids = []
        for sublist in coordinate_lists:
            if not sublist:
                centroids.append((0, 0))
                continue
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
            cv2.line(updated_image, coordinates[i], coordinates[i + 1], 
                     color=(0, 255, 0), thickness=2)
        return updated_image
    
    def process_video(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                height, width, _ = frame.shape
                road_path = np.zeros_like(frame)
                edges = self._gpu_preprocess_frame(frame)
                bottom_pixels = self._find_bottom_white_pixels(edges)
                filtered_bottom_pixels = self._filter_pixels(bottom_pixels)
                sections = self._split_coordinates(filtered_bottom_pixels)
                section_centroids = self._calculate_centroids(sections)
                traced_track = self._connect_coordinates(frame, bottom_pixels)

                for centroid in section_centroids:
                    cv2.circle(traced_track, centroid, 4, (255, 0, 0), -1)
                
                second_phase = []
                
                limit_y = 250

                for centroid in section_centroids:
                    cen_x, cen_y = centroid[0], centroid[-1]
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

                traced_track2 = fit_and_visualize_lines(third_phase_centroid, width, height)

                cv2.imshow('Track Frame', traced_track)
                cv2.imshow('Track Frame2', traced_track2)
                cv2.imshow('Edge Detection', edges)
                cv2.imshow('road_path', road_path)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    video_path = ''
    tracker = VideoTracker(video_path)
    tracker.process_video()

if __name__ == "__main__":
    main()
