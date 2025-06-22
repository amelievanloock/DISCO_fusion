import time
from pathlib import Path
from enum import Flag, auto
from typing import Optional, Dict, List, Set
import numpy as np
import cv2

from core.frame_management import VideoProcessor
from core.vehicle_detection import VehicleDetector, VehicleDetection
from core.license_plate_detection import LicensePlateDetector, LicensePlateDetection
from core.tracking import VehicleTracker
from utils.utils import crop_vehicle, create_output_path, save_config


VALID_STEPS = {'detect', 'plate', 'track'}

class UnifiedProcessor:
    def __init__(
        self,
        vehicle_detection_model_path: str,
        lp_detection_model_path: str,
        reid_model_path: str,
        strongsort_model_path: str,
        batch_size: int = 32,
        target_height: Optional[int] = 1080,
        device: str = 'cuda',
        vehicle_conf_threshold: float = 0.5,
        lp_conf_threshold: float = 0.5,
        video_path: str = None,
        output_path: str = None,
    ):
        """
        Initialize the processing pipeline.

        Args:
            detection_model_path: Path to YOLO detection model
            reid_model_path: Path to ReID model
            strongsort_model_path: Path to StrongSORT model
            batch_size: Number of frames to process at once
            target_height: Height to resize frames to (None for no resize)
            device: Device for model inference
            conf_threshold: Detection confidence threshold
        """
        self.video_path = video_path
        self.output_path = output_path
        self.vehicle_detection_model_path = vehicle_detection_model_path
        self.lp_detection_model_path = lp_detection_model_path
        self.reid_model_path = reid_model_path
        self.strongsort_model_path = strongsort_model_path
        self.start_time = None
        self.batch_size = batch_size
        self.total_frames = 0
        self.frames_processed = 0

        try:
            self.video_processor = VideoProcessor(
            video_path=video_path,
            buffer_size=batch_size,
            target_height=target_height
            )
            self.output_path = output_path
            self.batch_size = batch_size
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {e}")
        
    def _initialize_models(self, steps: Set[str], device: str,
                         vehicle_conf_threshold: float, lp_conf_threshold: float):
        """Initialize only the models needed for requested steps."""
        if 'detect' in steps:
            self.vehicle_detector = VehicleDetector(
                video_processor=self.video_processor,
                model_path=self.vehicle_detection_model_path,
                device=device,
                conf_threshold=vehicle_conf_threshold
            )
            
        if 'plate' in steps:
            self.lp_detector = LicensePlateDetector(
                video_processor=self.video_processor,
                model_path=self.lp_detection_model_path,
                device=device,
                conf_threshold=lp_conf_threshold
            )
            
        if 'track' in steps:
            self.tracker = VehicleTracker(
                reid_weights_path=self.reid_model_path,
                strongsort_weights_path=self.strongsort_model_path,
                device=device,
                video_processor=self.video_processor,
            )    

    def process_video(
        self, 
        steps: Set[str],
        vehicle_detections_path: Optional[Path] = None,
        plate_detections_path: Optional[Path] = None
    ) -> None:
        """Process video with configurable steps."""
        files = {}
        try:
            # Validate step # Validate steps
            if not steps:
                raise ValueError("No processing steps specified. Valid steps are: detect, plate, track")
            
            invalid_steps = steps - VALID_STEPS
            if invalid_steps:
                raise ValueError(f"Invalid steps: {invalid_steps}. Valid steps are: {VALID_STEPS}")
                
            # Validate dependencies
            if 'plate' in steps and 'detect' not in steps and not vehicle_detections_path:
                raise ValueError("Vehicle detections required for plate detection")
                
            if 'track' in steps and 'detect' not in steps and not vehicle_detections_path:
                raise ValueError("Vehicle detections required for tracking")
            
            # Initialize models before processing
            self._initialize_models(
                steps=steps,
                device='cuda',
                vehicle_conf_threshold=0.5,
                lp_conf_threshold=0.5
            )

            self.start_time = time.time()
            self.total_frames = self.video_processor.total_frames
            self.frames_processed = 0
            
            output_path = create_output_path(self.output_path, self.video_path)
            debug_dir = Path("/home/usuaris/imatge/amelie.van.loock/DISCO-opt/debug")
            save_config(self, output_path.parent)
            
            files = self._setup_output_files(output_path, steps)
            
            while self.frames_processed < self.total_frames:
                batch_start = time.time()
                
                frames_data = self.video_processor.process_frame_range(
                    self.frames_processed,
                    min(self.frames_processed + self.batch_size, self.total_frames)
                )
                
                if not frames_data:
                    print(f"Warning: No frames read at position {self.frames_processed}")
                    break
                
                
                    
                    
                # Vehicle Detection
                if 'detect' in steps:
                    vehicle_detections = self.vehicle_detector.process_batch(frames_data)
                    self._save_vehicle_detections(files['detect'], vehicle_detections)
                else:
                    vehicle_detections = self._load_vehicle_detections(
                        vehicle_detections_path,
                        [fd.frame_idx for fd in frames_data]
                    )
                vehicle_count = 0
                plate_count = 0
                
                for frame_data in frames_data:
                    frame_idx = frame_data.frame_idx
                    frame = frame_data.frame
                    frame_vehicle_dets = vehicle_detections.get(frame_idx, [])
                    if not frame_vehicle_dets:
                        continue
                
                    # --- License Plate Detection (original) ---
                    plate_detections = {}
                    if 'plate' in steps:
                        # crop each vehicle from the resized frame
                        vehicle_crops = [
                            crop_vehicle(frame, det.bbox)
                            for det in frame_vehicle_dets
                        ]
                
                        # debug: dump each vehicle crop (only first 30)
                        for i, det in enumerate(frame_vehicle_dets):
                            if vehicle_count >= 30:
                                break
                            print(f"[DEBUG][Frame {frame_idx}] Vehicle #{i}: bbox={det.bbox}, conf={det.confidence:.3f}")
                            cv2.imwrite(str(debug_dir / f"vehicle_{frame_idx:06d}_{i:02d}.png"), vehicle_crops[i])
                            vehicle_count += 1
                
                        # run your plate detector on those crops
                        plate_detections = self.lp_detector.detect_batch(vehicle_crops, frame_vehicle_dets)
                
                        # debug: dump each plate crop (only first 30)
                        for i, lp_det in enumerate(plate_detections.values()):
                            if plate_count >= 30:
                                break
                            veh_det = lp_det.vehicle_det
                            coords = lp_det.bbox  # in vehicle-crop coords
                            x1, y1, x2, y2 = map(int, coords)
                            print(
                                f"[DEBUG][Frame {frame_idx}] Plate on vehicle "
                                f"(frame {veh_det.frame_idx}): bbox={coords}, conf={lp_det.confidence:.3f}"
                            )
                
                            plate_crop = vehicle_crops[
                                frame_vehicle_dets.index(veh_det)
                            ][y1:y2, x1:x2]
                            cv2.imwrite(str(debug_dir / f"plate_{frame_idx:06d}_{i:02d}.png"), plate_crop)
                            plate_count += 1
          
    
                        # write plate detections to CSV
                        if 'plate' in files:
                            self._save_plate_detections(
                                files['plate'], frame_idx, plate_detections
                            )
    
                    # --- Tracking ---
                    if 'track' in steps:
                        tracks = self.tracker.update(frame, frame_vehicle_dets, frame_idx)
                        self._write_tracking_results(
                            files['track'], frame_idx, tracks, plate_detections
                        )
    
                batch_frames = len(frames_data)
                self.frames_processed += batch_frames
                batch_time = time.time() - batch_start
                self._log_progress(batch_frames, batch_time)
    
            self.print_summary()
    
        finally:
            for f in files.values():
                f.close()
            self.video_processor.clear()

    def _setup_output_files(self, output_path: Path, steps: Set[str]) -> Dict:
        """Initialize output files based on enabled steps."""
        files = {}
        
        if 'detect' in steps:
            files['detect'] = open(output_path.parent / "vehicle_detections.csv", 'w')
            files['detect'].write('frame,x1,y1,x2,y2,confidence\n')
            
        if 'plate' in steps:
            files['plate'] = open(output_path.parent / "plate_detections.csv", 'w')
            files['plate'].write('frame,vehicle_x1,vehicle_y1,vehicle_x2,vehicle_y2,'
                            'plate_x1,plate_y1,plate_x2,plate_y2,confidence\n')
            
        if 'track' in steps:
            files['track'] = open(output_path, 'w')
            files['track'].write('frame,vehicle_id,x1,y1,x2,y2,has_plate,'
                            'plate_x1,plate_y1,plate_x2,plate_y2,plate_conf\n')
        return files

    def _write_tracking_results(self, f, frame_idx: int, tracks: List, 
                          plate_detections: Dict) -> None:
        """Write tracking results with matched plate detections."""
        for track in tracks:
            track_bbox = tuple(int(x) for x in track.bbox)
            
            # Find matching plate
            matching_plate = None
            for plate_det in plate_detections.values():
                if tuple(int(x) for x in plate_det.vehicle_det.bbox) == track_bbox:
                    matching_plate = plate_det
                    break
            
            # Get plate coordinates and confidence only if we have a match
            plate_coords = matching_plate.absolute_coords if matching_plate is not None else [0, 0, 0, 0]
            plate_conf = matching_plate.confidence if matching_plate is not None else 0
            has_plate = 1 if matching_plate is not None else 0
            
            # Write results with safe access to values
            f.write(
                f"{frame_idx},{track.track_id},"
                f"{track.original_coords[0]:.2f},{track.original_coords[1]:.2f},"
                f"{track.original_coords[2]:.2f},{track.original_coords[3]:.2f},"
                f"{has_plate},"
                f"{plate_coords[0]:.2f},{plate_coords[1]:.2f},"
                f"{plate_coords[2]:.2f},{plate_coords[3]:.2f},"
                f"{plate_conf:.3f}\n"
            )

    def _save_vehicle_detections(self, f, vehicle_detections: Dict) -> None:
        """Save vehicle detections to file."""
        for frame_idx, detections in vehicle_detections.items():
            for det in detections:
                f.write(
                    f"{frame_idx},{det.bbox[0]:.2f},{det.bbox[1]:.2f},"
                    f"{det.bbox[2]:.2f},{det.bbox[3]:.2f},{det.confidence:.3f}\n"
                )
    def _load_vehicle_detections(self, detections_path: Path, frame_indices: List[int]) -> Dict[int, List[VehicleDetection]]:
        """
        Load vehicle detections from CSV file for specified frames.
        
        Args:
            detections_path: Path to vehicle detections CSV
            frame_indices: List of frame indices to load
            
        Returns:
            Dictionary mapping frame indices to lists of vehicle detections
        """
        detections = {}
        
        # Initialize empty lists for all requested frames
        for idx in frame_indices:
            detections[idx] = []
        
        with open(detections_path, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                frame_idx, x1, y1, x2, y2, conf = map(float, line.strip().split(','))
                frame_idx = int(frame_idx)
                
                # Only process frames we're interested in
                if frame_idx in frame_indices:
                    bbox = [x1, y1, x2, y2]
                    original_coords = self.video_processor.transform_coordinates(bbox, inverse=True)
                    
                    detection = VehicleDetection(
                        bbox=bbox,
                        confidence=conf,
                        class_id=2,  # Default to car class
                        frame_idx=frame_idx,
                        original_coords=original_coords
                    )
                    detections[frame_idx].append(detection)
        
        return detections
    def _save_plate_detections(self, f, frame_idx: int, plate_detections: Dict) -> None:
        """
        Save plate detections to file.
        
        Args:
            f: Output file handle
            frame_idx: Frame index
            plate_detections: Dictionary mapping vehicle detections to plate detections
        """
        for vehicle_det, plate_det in plate_detections.items():
        # Get absolute coordinates for plate bbox
            vehicle_det = plate_det.vehicle_det
            absolute_coords = plate_det.absolute_coords if hasattr(plate_det, 'absolute_coords') else plate_det.bbox
            f.write(
                f"{frame_idx},"
                f"{vehicle_det.bbox[0]:.2f},{vehicle_det.bbox[1]:.2f},"
                f"{vehicle_det.bbox[2]:.2f},{vehicle_det.bbox[3]:.2f},"
                f"{absolute_coords[0]:.2f},{absolute_coords[1]:.2f},"
                f"{absolute_coords[2]:.2f},{absolute_coords[3]:.2f},"
                f"{plate_det.confidence:.3f}\n"
            )
    
    def _load_plate_detections(self, detections_path: Path, frame_idx: int) -> Dict:
        """
        Load plate detections from CSV file for a specific frame.
        
        Args:
            detections_path: Path to plate detections CSV
            frame_idx: Frame index to load
            
        Returns:
            Dictionary mapping vehicle detections to plate detections
        """
        plate_detections = {}
        
        with open(detections_path, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                fields = line.strip().split(',')
                file_frame_idx = int(fields[0])
                
                # Only process requested frame
                if file_frame_idx == frame_idx:
                    # Parse vehicle bbox
                    vehicle_bbox = [float(x) for x in fields[1:5]]
                    # Parse plate bbox
                    plate_bbox = [float(x) for x in fields[5:9]]
                    confidence = float(fields[9])
                    
                    # Create vehicle detection object
                    vehicle_det = VehicleDetection(
                        bbox=vehicle_bbox,
                        confidence=1.0,  # Original confidence not stored
                        class_id=2,      # Default to car class
                        frame_idx=frame_idx,
                        original_coords=self.video_processor.transform_coordinates(vehicle_bbox, inverse=True)
                    )
                    
                    # Create plate detection object
                    plate_det = LicensePlateDetection(
                        bbox=plate_bbox,
                        confidence=confidence,
                        vehicle_det=vehicle_det,
                        absolute_coords=self.video_processor.transform_coordinates(plate_bbox, inverse=True)
                    )
                    
                    # Store using vehicle detection as key
                    plate_detections[hash(str(vehicle_bbox))] = plate_det
        
        return plate_detections

    def _log_progress(self, batch_frames: int, batch_time: float) -> None:
        """Log processing progress with current FPS."""
        fps = batch_frames / batch_time
        progress = (self.frames_processed / self.total_frames) * 100
        elapsed = time.time() - self.start_time
        print(f"Processed frames {self.frames_processed}/{self.total_frames} "
              f"({progress:.1f}%) at {fps:.2f} FPS | "
              f"Elapsed: {elapsed:.0f}s")

    def print_summary(self) -> None:
        """Print processing summary with timing and detection/tracking statistics."""
        total_time = time.time() - self.start_time
        fps = self.video_processor.buffer.fps 
        video_duration = self.total_frames / fps  
        processing_ratio = total_time / video_duration
        
        print("\nProcessing Summary")
        print("=================")
        print(f"Video Information:")
        print(f"  Duration: {video_duration:.2f} seconds ({video_duration/60:.1f} minutes)")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Video frame rate: {fps:.2f} FPS")
        
        print("\nProcessing Performance:")
        print(f"  Processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Average processing speed: {self.frames_processed/total_time:.2f} FPS")
        print(f"  Real-time ratio: {processing_ratio:.2f}x")