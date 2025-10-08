#!/usr/bin/env python3
"""
Dataset Construction Script

Constructs video hallucination dataset by combining target videos with event clips
based on metadata.
"""

import argparse
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Video processing imports
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm


def load_metadata(metadata_path):
    """Load metadata from JSON file"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def combine_videos(source_video_path, event_video_path, position, insert_timestamp=None):
    """Combine source video with event video based on position"""
    source_clip = VideoFileClip(str(source_video_path))
    event_clip = VideoFileClip(str(event_video_path))
    
    # Fix FPS mismatch - set event clip to match source clip FPS
    target_fps = source_clip.fps
    if event_clip.fps != target_fps:
        event_clip = event_clip.set_fps(target_fps)
    
    # Remove audio from all clips for consistency
    if source_clip.audio is not None:
        source_clip = source_clip.without_audio()
    if event_clip.audio is not None:
        event_clip = event_clip.without_audio()
    
    if position == "start":
        final_clip = concatenate_videoclips([event_clip, source_clip], method="compose")
    elif position == "end":
        final_clip = concatenate_videoclips([source_clip, event_clip], method="compose")
    elif position == "center" and insert_timestamp is not None:
        before = source_clip.subclip(0, insert_timestamp)
        after = source_clip.subclip(insert_timestamp, source_clip.duration)
        final_clip = concatenate_videoclips([before, event_clip, after], method="compose")
    else:
        raise ValueError(f"Invalid position: {position}")
    
    # Ensure final clip has consistent FPS
    final_clip = final_clip.set_fps(target_fps)
    
    return final_clip


def process_single_recipe(recipe, event_clip_dir, target_video_dir, output_dir):
    """Process a single recipe to create combined video"""
    try:
        target_video_id = recipe['target_id']
        event_clip_id = recipe['event_id']
        insert_timestamp = recipe['insert_timestamp']
        similarity = recipe['similarity']
        position = recipe['position']
        
        target_video_path = target_video_dir / f"{target_video_id}.mp4"
        event_clip_path = event_clip_dir / f"{event_clip_id}.mp4"
        
        if not target_video_path.exists():
            logging.warning(f"Target video not found: {target_video_path}")
            return False
        if not event_clip_path.exists():
            logging.warning(f"Event clip not found: {event_clip_path}")
            return False
        
        combined_video = combine_videos(target_video_path, event_clip_path, position, insert_timestamp)
        
        output_filename = f"{target_video_id}_{event_clip_id}_{similarity}_{position}.mp4"
        output_path = output_dir / output_filename
        
        # Skip if already exists
        if output_path.exists():
            logging.info(f"Skipping existing file: {output_filename}")
            combined_video.close()
            return True
        
        combined_video.write_videofile(str(output_path), audio=False, verbose=False, logger=None)
        
        logging.info(f"Successfully created: {output_filename}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing recipe {recipe.get('target_id', 'unknown')}: {e}")
        return False


def process_recipe_wrapper(args):
    """Wrapper function for thread pool processing"""
    recipe, event_clip_dir, target_video_dir, output_dir = args
    return process_single_recipe(recipe, event_clip_dir, target_video_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Construct video hallucination dataset")
    parser.add_argument("--metadata", default="data/noah/metadata.json", help="Path to metadata JSON file")
    parser.add_argument("--event-clip-dir", default="data/activitynet_clips",
                        help="Directory containing event clips (default: data/activitynet_clips)")
    parser.add_argument("--target-video-dir", default="data/activitynet_videos",
                        help="Directory containing original video files (.mp4) (default: data/activitynet_videos)")
    parser.add_argument("--output-dir", default="data/noah/videos",
                        help="Output directory for combined videos (default: data/noah/videos)")
    parser.add_argument("--max-workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    parser.add_argument("--log-file", default="data/logs/build_dataset_log.txt",
                        help="Log file path (default: data/logs/build_dataset_log.txt)")
    
    args = parser.parse_args()
    
    # Create log directory if needed
    log_dir = Path(args.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file)
        ]
    )
    
    logging.info("Loading metadata...")
    metadata = load_metadata(args.metadata)
    
    event_clip_dir = Path(args.event_clip_dir)
    target_video_dir = Path(args.target_video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Processing {len(metadata)} items...")
    
    success_count = 0
    failed_count = 0
    
    recipe_args = [(recipe, event_clip_dir, target_video_dir, output_dir) for recipe in metadata]
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_recipe = {executor.submit(process_recipe_wrapper, args): args[0] for args in recipe_args}
        
        with tqdm(total=len(metadata), desc="Processing videos", unit="video") as pbar:
            for future in as_completed(future_to_recipe):
                recipe = future_to_recipe[future]
                target_id = recipe.get('target_id', 'unknown')
                
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logging.error(f"Thread error processing {target_id}: {e}")
                    failed_count += 1
                
                pbar.set_postfix(success=success_count, failed=failed_count, workers=args.max_workers)
                pbar.update(1)
                
                if (success_count + failed_count) % 100 == 0:
                    total_processed = success_count + failed_count
                    logging.info(f"Progress: {total_processed}/{len(metadata)} ({total_processed/len(metadata)*100:.1f}%)")
    
    logging.info("Processing complete!")
    logging.info(f"Success: {success_count}")
    logging.info(f"Failed: {failed_count}")
    
    total_count = success_count + failed_count
    if total_count > 0:
        logging.info(f"Success rate: {success_count/total_count*100:.1f}%")


if __name__ == "__main__":
    main()
