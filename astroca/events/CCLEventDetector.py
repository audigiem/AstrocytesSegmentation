import numpy as np
from scipy import ndimage
from typing import Tuple, List
import time
import os
from skimage.measure import label
import matplotlib.pyplot as plt
from astroca.tools.exportData import export_data


class CCLEventDetector:
    """
    Event Detector using Connected Component Labeling (CCL) for 4D data (time, depth, height, width).
    """

    def __init__(self, av_data: np.ndarray,
                 min_size_3d: int = 10,
                 min_duration: int = 3,
                 structure_3d: np.ndarray = None):
        """
        Initialize the detector with 4D data and parameters.

        Args:
            av_data: 4D numpy array (T, Z, Y, X) of active voxels
            min_size_3d: Minimum size of 3D components to consider
            min_duration: Minimum duration in time frames
            structure_3d: Connectivity structure for 3D space (default: 26-connectivity)
        """
        self.av_data = av_data.astype(np.float32)
        self.time_len, self.depth, self.height, self.width = av_data.shape
        self.min_size_3d = min_size_3d
        self.min_duration = min_duration

        # Default connectivity structure (26-connectivity in 3D)
        self.structure_3d = np.ones((3, 3, 3), dtype=bool) if structure_3d is None else structure_3d

        # Results storage
        self.labels_4d = None
        self.event_stats = []

    def detect_events(self) -> np.ndarray:
        """
        Main detection method using connected component analysis.

        Returns:
            4D array with labeled events
        """
        start_time = time.time()

        # Binarize the data
        binary_data = self.av_data > 0

        # First pass: 3D CCL for each time frame
        time_labels = np.zeros_like(self.av_data, dtype=np.int32)

        for t in range(self.time_len):
            # Use ndimage.label instead of skimage.measure.label for structure parameter
            labeled_frame, num_features = ndimage.label(binary_data[t], structure=self.structure_3d)
            time_labels[t] = labeled_frame + (time_labels.max() if t > 0 else 0)

        # Second pass: Temporal connection between 3D components
        self.labels_4d = self._connect_temporally(time_labels)

        # Filter small events
        self._filter_events()

        print(f"Event detection completed in {time.time() - start_time:.2f} seconds")
        return self.labels_4d

    def _connect_temporally(self, time_labels: np.ndarray) -> np.ndarray:
        """
        Connect 3D components across time frames.

        Args:
            time_labels: 4D array with labeled 3D components per frame

        Returns:
            4D array with temporally connected components
        """
        # Create a mapping between labels in consecutive frames
        label_mapping = {}
        current_max = 1

        # Initialize output with first frame
        connected_labels = np.zeros_like(time_labels)
        connected_labels[0] = time_labels[0]

        for t in range(1, self.time_len):
            current_frame = time_labels[t]
            prev_frame = connected_labels[t - 1]

            # Find overlapping components between frames
            overlap_labels = {}

            # Only consider voxels that are active in both frames
            overlap_mask = (current_frame > 0) & (prev_frame > 0)
            current_ids = current_frame[overlap_mask]
            prev_ids = prev_frame[overlap_mask]

            # Create mapping based on overlap
            for curr, prev in zip(current_ids, prev_ids):
                if curr not in overlap_labels:
                    overlap_labels[curr] = []
                overlap_labels[curr].append(prev)

            # Determine the best mapping for each current label
            for curr in overlap_labels:
                # Find the most common previous label
                prev_counts = {}
                for p in overlap_labels[curr]:
                    prev_counts[p] = prev_counts.get(p, 0) + 1

                if prev_counts:
                    best_prev = max(prev_counts, key=prev_counts.get)
                    label_mapping[curr] = best_prev

            # Apply the mapping
            new_frame = np.zeros_like(current_frame)
            for curr in np.unique(current_frame):
                if curr == 0:
                    continue
                if curr in label_mapping:
                    new_frame[current_frame == curr] = label_mapping[curr]
                else:
                    new_frame[current_frame == curr] = current_max
                    label_mapping[curr] = current_max
                    current_max += 1

            connected_labels[t] = new_frame

        return connected_labels

    def _filter_events(self):
        """
        Filter events based on size and duration criteria.
        """
        unique_labels = np.unique(self.labels_4d)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        valid_labels = []
        self.event_stats = []

        for label_id in unique_labels:
            # Create mask for this event
            mask = self.labels_4d == label_id

            # Calculate 3D size per time frame
            sizes_per_frame = [np.sum(np.any(mask[t], axis=(1, 2))) for t in range(self.time_len)]
            max_3d_size = max(sizes_per_frame)

            # Calculate duration
            active_frames = [s > 0 for s in sizes_per_frame]
            duration = np.sum(active_frames)

            if max_3d_size >= self.min_size_3d and duration >= self.min_duration:
                valid_labels.append(label_id)
                self.event_stats.append({
                    'id': label_id,
                    'max_size': max_3d_size,
                    'duration': duration,
                    'start_frame': np.argmax(active_frames),
                    'end_frame': self.time_len - 1 - np.argmax(active_frames[::-1])
                })
            else:
                # Remove small/short events
                self.labels_4d[mask] = 0

        # Re-label consecutively
        self._relabel_events(valid_labels)

    def _relabel_events(self, valid_labels: List[int]):
        """
        Re-label events to have consecutive IDs.
        """
        if not valid_labels:
            return

        # Create mapping from old to new labels
        label_map = {old: new + 1 for new, old in enumerate(sorted(valid_labels))}
        label_map[0] = 0  # Background remains 0

        # Apply mapping
        for old, new in label_map.items():
            self.labels_4d[self.labels_4d == old] = new

        # Update stats with new IDs
        for i, stat in enumerate(self.event_stats):
            stat['id'] = i + 1

    def get_event_statistics(self) -> List[dict]:
        """
        Get statistics for detected events.

        Returns:
            List of dictionaries with event statistics
        """
        return self.event_stats

    def visualize_event(self, event_id: int, max_project: bool = True):
        """
        Visualize a detected event.

        Args:
            event_id: ID of event to visualize
            max_project: Whether to max-project over Z axis
        """
        if self.labels_4d is None:
            raise ValueError("Run detect_events() first")

        mask = self.labels_4d == event_id
        event_data = self.av_data * mask

        # Find time range
        stats = next(s for s in self.event_stats if s['id'] == event_id)
        t_start, t_end = stats['start_frame'], stats['end_frame']

        # Create visualization
        n_frames = t_end - t_start + 1
        fig, axes = plt.subplots(1, n_frames, figsize=(15, 5)) if n_frames > 1 else (plt.figure(figsize=(5, 5)),
                                                                                     [plt.gca()])

        for t in range(t_start, t_end + 1):
            frame = event_data[t]
            if max_project:
                frame = np.max(frame, axis=0)

            ax = axes[t - t_start]
            ax.imshow(frame, cmap='hot')
            ax.set_title(f"Frame {t}")
            ax.axis('off')

        plt.suptitle(f"Event {event_id} (Size: {stats['max_size']}, Duration: {stats['duration']})")
        plt.tight_layout()
        plt.show()


def detect_events_4d(av_data: np.ndarray,
                     params: dict = None) -> Tuple[np.ndarray, List[dict]]:
    """
    Convenience function for event detection.

    Args:
        av_data: 4D input data (T, Z, Y, X)
        params: Dictionary of parameters (min_size_3d, min_duration)

    Returns:
        Tuple of (labeled_events, event_statistics)
    """
    if params is None:
        params = {
            'min_size_3d': 10,
            'min_duration': 1
        }
        min_size_3d = params['min_size_3d']
        min_duration = params['min_duration']
        output_dir = None
        save_results = False
    else:
        min_size_3d = params['events_extraction']['threshold_size_3d']
        min_duration = 1
        output_dir = params['paths']['output_dir']
        save_results = int(params['files']['save_results']) == 1 if 'save_results' in params['files'] else False

    detector = CCLEventDetector(av_data, min_size_3d, min_duration)

    labels = detector.detect_events()
    stats = detector.get_event_statistics()

    if save_results:
        if output_dir is None:
            raise ValueError("Output directory must be specified in params['paths']['output_dir']")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        export_data(labels, output_dir, export_as_single_tif=True, file_name="events_labels.tif")

    return labels, stats


def test_with_synthetic_data():
    """Test function with synthetic data"""
    print("Testing with synthetic data...")

    # Create a 4D volume with two events
    shape = (10, 20, 50, 50)  # T, Z, Y, X
    data = np.zeros(shape, dtype=np.float32)

    # First event: small but long duration
    data[1:8, 5:10, 10:20, 10:20] = 0.5
    data[2:7, 6:9, 12:18, 12:18] = 1.0

    # Second event: large but short duration
    data[5:7, 12:18, 30:45, 30:45] = 1.0

    # Add some noise
    data += np.random.normal(0, 0.1, size=shape)
    data = np.clip(data, 0, 1)

    # Detect events
    labels, stats = detect_events_4d(data, {
        'min_size_3d': 5,
        'min_duration': 2
    })

    print(f"Detected {len(stats)} events:")
    for stat in stats:
        print(f"Event {stat['id']}: Size={stat['max_size']}, Duration={stat['duration']}")

    # Visualize first event
    if len(stats) > 0:
        detector = CCLEventDetector(data)
        detector.labels_4d = labels
        detector.event_stats = stats
        detector.visualize_event(1)

    return labels, stats


if __name__ == "__main__":
    test_with_synthetic_data()