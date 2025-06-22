import argparse
from pathlib import Path
from pipeline import UnifiedProcessor
from utils.utils import load_config 
from typing import Set


def parse_steps(steps_str: str) -> Set[str]:
    """Convert comma-separated string of steps to set of steps."""
    if not steps_str:
        raise ValueError("No steps specified")
    return {step.lower() for step in steps_str.split(',')}

def main(args):
    config = load_config(args.config)
    steps = parse_steps(args.steps)
    
    processor = UnifiedProcessor(
        vehicle_detection_model_path=config['models']['detection']['vehicle_path'],
        lp_detection_model_path=config['models']['detection']['lp_path'],
        reid_model_path=config['models']['tracking']['reid_path'],
        strongsort_model_path=config['models']['tracking']['strongsort_path'],
        batch_size=config['processing']['batch_size'],
        target_height=config['processing']['target_height'],
        vehicle_conf_threshold=config['processing']['vehicle_confidence_threshold'],
        lp_conf_threshold=config['processing']['lp_confidence_threshold'],
        video_path=args.video_path,
        output_path=args.output_path,
    )
    
    processor.process_video(
        steps=steps,
        vehicle_detections_path=args.vehicle_detections,
        plate_detections_path=args.plate_detections
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle detection and tracking pipeline")
    parser.add_argument('--video_path', required=True, help='Path to input video')
    parser.add_argument('--output_path', required=True, help='Path for results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--steps', help='Comma-separated list of steps to run (DETECT,PLATE,TRACK). Default: all')
    parser.add_argument('--vehicle_detections', help='Path to vehicle detections CSV (required for PLATE/TRACK without DETECT)')
    parser.add_argument('--plate_detections', help='Path to plate detections CSV (optional for TRACK)')
    parser.add_argument('--use_plates', action='store_true', help='Use plate detections in tracking step')
    
    args = parser.parse_args()
    main(args)


    """ 
# Run full pipeline
python main.py --video_path /mnt/gpid08/datasets/DISCO/fragments/video_5.mp4 --output_path results/ --config config.yaml --steps detect,plate,track

# Run only vehicle detection
python main.py --video_path /mnt/gpid08/datasets/DISCO/fragments/video_5.mp4 --output_path results/ --config config.yaml --steps detect

# Run plate detection using existing vehicle detections 
python main.py --video_path /mnt/gpid08/datasets/DISCO/fragments/video_5.mp4 --output_path results/ --config config.yaml --steps plate --vehicle_detections results/video_5_*/vehicle_detections.csv

# Run tracking with existing detections and plates
python main.py --video_path /mnt/gpid08/datasets/DISCO/fragments/video_5.mp4 --output_path results/ --config config.yaml --steps track --vehicle_detections results/video_5_*/vehicle_detections.csv --plate_detections results/video_5_*/plate_detections.csv

# Run detection and tracking, skip plate detection
python main.py --video_path /mnt/gpid08/datasets/DISCO/fragments/video_5.mp4 --output_path results/ --config config.yaml --steps detect,track
    
    
use relative path when importing detections from csv
    """