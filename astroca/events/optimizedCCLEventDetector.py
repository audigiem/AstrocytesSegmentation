import numpy as np
from scipy import ndimage
from typing import Tuple, List, Dict
import time
import os
from skimage.measure import label
import matplotlib.pyplot as plt
from scipy.signal import correlate


class CCLEventDetector:
    """
    Event Detector using Connected Component Labeling with enhanced temporal connection.
    """

    def __init__(self, av_data: np.ndarray,
                 min_size_3d: int = 10,
                 min_duration: int = 3,
                 corr_threshold: float = 0.5,
                 structure_3d: np.ndarray = None):
        """
        Initialize the detector with enhanced temporal connection.

        Args:
            av_data: 4D numpy array (T, Z, Y, X) of active voxels
            min_size_3d: Minimum size of 3D components to consider
            min_duration: Minimum duration in time frames
            corr_threshold: Correlation threshold for temporal connection
            structure_3d: Connectivity structure for 3D space
        """
        self.av_data = av_data.astype(np.float32)
        self.time_len, self.depth, self.height, self.width = av_data.shape
        self.min_size_3d = min_size_3d
        self.min_duration = min_duration
        self.corr_threshold = corr_threshold

        # Default connectivity structure (26-connectivity in 3D)
        self.structure_3d = np.ones((3, 3, 3), dtype=bool) if structure_3d is None else structure_3d

        # Results storage
        self.labels_4d = None
        self.event_stats = []
        self.patterns = {}  # Store temporal patterns for each component

    def detect_events(self) -> np.ndarray:
        """
        Main detection method with enhanced temporal connection.
        """
        start_time = time.time()

        # Binarize the data
        binary_data = self.av_data > 0

        # First pass: 3D CCL for each time frame
        time_labels = np.zeros_like(self.av_data, dtype=np.int32)

        print("Labeling 3D components per frame...")
        for t in range(self.time_len):
            labeled_frame, num_features = label(binary_data[t], structure=self.structure_3d)
            time_labels[t] = labeled_frame + (time_labels.max() if t > 0 else 0)

        # Extract temporal patterns for each 3D component
        print("Extracting temporal patterns...")
        self._extract_temporal_patterns(time_labels)

        # Connect components temporally using pattern correlation
        print("Connecting components temporally...")
        self.labels_4d = self._connect_with_correlation(time_labels)

        # Filter small events
        print("Filtering events...")
        self._filter_events()

        print(f"Event detection completed in {time.time() - start_time:.2f} seconds")
        return self.labels_4d

    def _extract_temporal_patterns(self, time_labels: np.ndarray):
        """
        Extract temporal patterns for each 3D component.
        """
        self.patterns = {}

        for t in range(self.time_len):
            for label_id in np.unique(time_labels[t]):
                if label_id == 0:
                    continue

                # Get mask for this component
                mask = time_labels[t] == label_id

                # Calculate mean intensity profile across the component
                intensity_profile = np.mean(self.av_data[:, mask], axis=1)

                # Find start and end of the pattern
                start = t
                while start > 0 and intensity_profile[start - 1] > 0:
                    start -= 1

                end = t + 1
                while end < self.time_len and intensity_profile[end] > 0:
                    end += 1

                pattern = intensity_profile[start:end]
                self.patterns[(t, label_id)] = pattern

    def _connect_with_correlation(self, time_labels: np.ndarray) -> np.ndarray:
        """
        Connect components across time using pattern correlation.
        """
        connected_labels = np.zeros_like(time_labels)
        current_max = 1
        label_mapping = {}  # Maps old labels to new connected labels

        # Initialize with first frame
        connected_labels[0] = time_labels[0]
        for label_id in np.unique(time_labels[0]):
            if label_id != 0:
                label_mapping[(0, label_id)] = current_max
                current_max += 1

        for t in range(1, self.time_len):
            print(f"Processing frame {t + 1}/{self.time_len}", end='\r')

            # Get current and previous frame labels
            current_labels = time_labels[t]
            prev_labels = connected_labels[t - 1]

            # Find overlapping components between frames
            overlap_mask = (current_labels > 0) & (prev_labels > 0)
            current_ids = current_labels[overlap_mask]
            prev_ids = prev_labels[overlap_mask]

            # Create mapping based on correlation
            correlation_mapping = {}

            for curr_id in np.unique(current_labels):
                if curr_id == 0:
                    continue

                # Get current component's pattern
                if (t, curr_id) not in self.patterns:
                    continue
                curr_pattern = self.patterns[(t, curr_id)]

                best_corr = -1
                best_prev_id = -1

                # Check all overlapping previous components
                for prev_id in np.unique(prev_labels):
                    if prev_id == 0:
                        continue

                    # Get previous component's pattern
                    if (t - 1, prev_id) not in self.patterns:
                        continue
                    prev_pattern = self.patterns[(t - 1, prev_id)]

                    # Compute normalized cross-correlation
                    corr = correlate(curr_pattern, prev_pattern, mode='full', method='auto')
                    norm = np.sqrt(np.sum(curr_pattern ** 2) * np.sqrt(np.sum(prev_pattern ** 2)))
                    if norm > 0:
                        corr = corr / norm
                    max_corr = np.max(corr)
                    if max_corr > best_corr:
                        best_corr = max_corr
                    best_prev_id = prev_id

                    # If correlation is above threshold, create mapping
                    if best_corr >= self.corr_threshold:
                        correlation_mapping[curr_id] = best_prev_id

                    # Apply the mapping
                    new_frame = np.zeros_like(current_labels)
                    for curr_id in np.unique(current_labels):
                        if curr_id == 0:
                            continue

                if curr_id in correlation_mapping:
                    # Get the new label from the previous frame
                    new_id = label_mapping.get((t - 1, correlation_mapping[curr_id]), current_max)
                    new_frame[current_labels == curr_id] = new_id
                    label_mapping[(t, curr_id)] = new_id
                else:
                    # Assign new label
                    new_frame[current_labels == curr_id] = current_max
                    label_mapping[(t, curr_id)] = current_max
                    current_max += 1

            connected_labels[t] = new_frame

        print("\nTemporal connection completed.")
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
        """
        return self.event_stats

    def visualize_event(self, event_id: int, max_project: bool = True):
        """
        Visualize a detected event.
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
        fig, axes = plt.subplots(1, n_frames, figsize=(15, 5)) if n_frames > 1 else (
        plt.figure(figsize=(5, 5)), [plt.gca()])

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
                     params: dict = None,
                     output_dir: str = None) -> Tuple[np.ndarray, List[dict]]:
    """
    Convenience function for event detection.

    Args:
        av_data: 4D input data (T, Z, Y, X)
        params: Dictionary of parameters
        output_dir: If provided, save results to this directory

    Returns:
        Tuple of (labeled_events, event_statistics)
    """
    default_params = {
        'min_size_3d': 10,
        'min_duration': 3,
        'corr_threshold': 0.5
    }
    if params is not None:
        default_params.update(params)

    detector = CCLEventDetector(
        av_data,
        min_size_3d=default_params['min_size_3d'],
        min_duration=default_params['min_duration'],
        corr_threshold=default_params['corr_threshold']
    )

    labels = detector.detect_events()
    stats = detector.get_event_statistics()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'event_labels.npy'), labels)

        # Save stats as text
        with open(os.path.join(output_dir, 'event_stats.txt'), 'w') as f:
            for stat in stats:
                f.write(f"Event {stat['id']}: Size={stat['max_size']}, Duration={stat['duration']}, "
                        f"Frames={stat['start_frame']}-{stat['end_frame']}\n")

    return labels, stats


def test_with_synthetic_data():
    """Test function with synthetic data"""
    print("Testing with synthetic data...")

    # Create a 4D volume with events
    shape = (15, 20, 50, 50)  # T, Z, Y, X
    data = np.zeros(shape, dtype=np.float32)

    # First event: small but long duration with specific pattern
    data[1:10, 5:10, 10:20, 10:20] = 0.5
    data[2:8, 6:9, 12:18, 12:18] = np.linspace(0.8, 1.2, 6).reshape(-1, 1, 1, 1)

    # Second event: large but short duration
    data[5:9, 12:18, 30:45, 30:45] = 1.0

    # Third event: similar spatial location but different temporal pattern
    data[8:12, 5:10, 15:25, 15:25] = np.linspace(1.2, 0.8, 4).reshape(-1, 1, 1, 1)

    # Add some noise
    data += np.random.normal(0, 0.1, size=shape)
    data = np.clip(data, 0, None)

    # Detect events
    labels, stats = detect_events_4d(data, {
        'min_size_3d': 5,
        'min_duration': 2,
        'corr_threshold': 0.4
    })

    print(f"Detected {len(stats)} events:")
    for stat in stats:
        print(f"Event {stat['id']}: Size={stat['max_size']}, Duration={stat['duration']}")

    # Visualize events
    if len(stats) > 0:
        detector = CCLEventDetector(data)
        detector.labels_4d = labels
        detector.event_stats = stats
        for i in range(min(3, len(stats))):
            detector.visualize_event(i + 1)

    return labels, stats


if __name__ == "__main__":
    test_with_synthetic_data()