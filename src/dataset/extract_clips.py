import json
import os
import argparse
from datetime import datetime
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_data(json_path):
    """Load JSON data from file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_unique_events(metadata_path):
    """
    Extract unique events from metadata_final.json.
    
    Args:
        metadata_path: Path to metadata_final.json file
        
    Returns:
        list: [{"event_id": "...", "event_timestamps": [start, end]}, ...]
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Remove duplicates: extract unique events using event_id as key
    unique_events_dict = {}
    for item in metadata:
        event_id = item['event_id']
        if event_id not in unique_events_dict:
            unique_events_dict[event_id] = {
                'event_id': event_id,
                'event_timestamps': item['event_timestamps']
            }
    
    unique_events = list(unique_events_dict.values())
    
    return unique_events

# Thread-safe logging lock
log_lock = threading.Lock()

def log_result(log_file, target_event_id, status, error_msg=None):
    """Write processing results to log file (thread-safe)."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if error_msg:
        log_entry = f"[{timestamp}] {target_event_id}: {status} - {error_msg}\n"
    else:
        log_entry = f"[{timestamp}] {target_event_id}: {status}\n"
    
    with log_lock:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(log_entry.strip())

def group_events_by_video(unique_events):
    """
    Group unique events by video_id.
    
    Args:
        unique_events: List of dicts with 'event_id' and 'event_timestamps'
        
    Returns:
        dict: {video_id: [(event_index, event_id, timestamps), ...]}
    """
    video_groups = defaultdict(list)
    
    for event in unique_events:
        event_id = event['event_id']
        timestamps = event['event_timestamps']
        
        # Extract video_id and event_index from event_id
        parts = event_id.rsplit('_', 1)
        if len(parts) != 2:
            continue
            
        video_id = parts[0]
        try:
            event_index = int(parts[1])
            video_groups[video_id].append((event_index, event_id, timestamps))
        except ValueError:
            continue
    
    # Sort events by index for each video
    for video_id in video_groups:
        video_groups[video_id].sort(key=lambda x: x[0])
    
    return dict(video_groups)

def process_single_video(video_id, events, event_clip_dir, target_video_dir, log_file):
    """
    Process all events for a single video (thread-safe).
    
    Args:
        video_id: Video ID
        events: List of (event_index, event_id, timestamps) tuples
        event_clip_dir: Directory to save extracted clips
        target_video_dir: Directory containing original video files
        log_file: Path to log file
    
    Returns:
        tuple: (extracted_count, failed_count, skipped_count)
    """
    extracted_count = 0
    failed_count = 0
    skipped_count = 0
    
    video_loaded = False
    video_clip = None
    
    try:
        for event_index, event_id, timestamps in events:
            output_filename = f"{event_id}.mp4"
            output_path = os.path.join(event_clip_dir, output_filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                log_result(log_file, event_id, "SKIP", "Already exists")
                skipped_count += 1
                continue
            
            # Load video if not already loaded
            if not video_loaded:
                original_video_path = os.path.join(target_video_dir, f"{video_id}.mp4")
                
                if not os.path.exists(original_video_path):
                    log_result(log_file, event_id, "FAIL", "Original video file not found")
                    failed_count += 1
                    break
                
                try:
                    video_clip = VideoFileClip(original_video_path)
                    video_loaded = True
                except Exception as e:
                    log_result(log_file, event_id, "FAIL", f"Failed to load video: {str(e)}")
                    failed_count += 1
                    break
            
            # Get timestamps from metadata
            start_time, end_time = timestamps
            
            # Adjust time to prevent duration overflow
            epsilon = 0.001
            original_end_time = end_time
            if end_time > video_clip.duration:
                end_time = video_clip.duration - epsilon
                log_result(log_file, event_id, "WARNING", f"Adjusted end_time from {original_end_time:.3f} to {end_time:.3f}")
            
            if end_time <= start_time:
                original_end_time_for_zero = end_time
                end_time = start_time + epsilon
                log_result(log_file, event_id, "WARNING", f"Fixed zero duration: start={start_time:.3f}, end from {original_end_time_for_zero:.3f} to {end_time:.3f}")
            
            # Extract and save clip
            try:
                clip = video_clip.subclip(start_time, end_time)
                clip.write_videofile(output_path, codec='libx264', audio=False, verbose=False, logger=None)
                clip.close()
                
                log_result(log_file, event_id, "EXTRACTED")
                extracted_count += 1
                
            except Exception as e:
                log_result(log_file, event_id, "EXTRACT_FAIL", str(e))
                failed_count += 1
                continue
                
    finally:
        # Release video memory
        if video_clip is not None:
            video_clip.close()
    
    return extracted_count, failed_count, skipped_count

def extract_target_event_clips(metadata_path, event_clip_dir, target_video_dir, log_file, max_workers=None):
    """
    Extract event clips from videos using metadata.
    Groups events by video_id and uses ThreadPool for parallel processing.
    
    Args:
        metadata_path: Path to metadata_final.json file
        event_clip_dir: Directory to save extracted event clips
        target_video_dir: Directory containing original video files
        log_file: Path to log file
        max_workers: Maximum number of worker threads (default: min(32, cpu_count()))
    """
    
    print("Loading metadata and extracting unique events...")
    unique_events = extract_unique_events(metadata_path)
    print(f"Total unique events to process: {len(unique_events)}")
    
    if not os.path.exists(event_clip_dir):
        os.makedirs(event_clip_dir)
    
    print("Grouping events by video ID...")
    video_groups = group_events_by_video(unique_events)
    print(f"Grouped into {len(video_groups)} unique videos")
    
    total_extracted = 0
    total_failed = 0
    total_skipped = 0
    
    if max_workers is None:
        max_workers = min(32, os.cpu_count())
    print(f"Using {max_workers} threads for parallel processing...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for video_id, events in video_groups.items():
            future = executor.submit(
                process_single_video, 
                video_id, events, event_clip_dir, target_video_dir, log_file
            )
            future_to_video[future] = video_id
        
        completed_videos = 0
        total_videos = len(video_groups)
        
        with tqdm(total=total_videos, desc="Processing videos") as pbar:
            for future in as_completed(future_to_video):
                video_id = future_to_video[future]
                try:
                    extracted, failed, skipped = future.result()
                    total_extracted += extracted
                    total_failed += failed
                    total_skipped += skipped
                    
                    completed_videos += 1
                    pbar.set_postfix({
                        'Completed': f"{completed_videos}/{total_videos}",
                        'Extracted': total_extracted,
                        'Failed': total_failed,
                        'Skipped': total_skipped
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    log_result(log_file, f"VIDEO_{video_id}", "THREAD_FAIL", f"Thread failed: {str(e)}")
                    total_failed += 1
                    completed_videos += 1
                    pbar.update(1)
    
    # Print final statistics
    total_processed = total_extracted + total_failed + total_skipped
    print(f"\n=== Final Statistics ===")
    print(f"Total processed: {total_processed}")
    print(f"Skipped (already exists): {total_skipped}")
    print(f"Extracted from original videos: {total_extracted}")
    print(f"Failed: {total_failed}")
    print(f"Threads used: {max_workers}")
    
    # Write statistics to log
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Final Statistics ===\n")
        f.write(f"Total processed: {total_processed}\n")
        f.write(f"Skipped (already exists): {total_skipped}\n")
        f.write(f"Extracted from original videos: {total_extracted}\n")
        f.write(f"Failed: {total_failed}\n")
        f.write(f"Threads used: {max_workers}\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract event clips from videos using metadata")
    
    parser.add_argument("--metadata", default="data/noah/metadata.json",
                        help="Path to metadata file (default: data/noah/metadata.json)")
    parser.add_argument("--event-clip-dir", default="data/activitynet_clips",
                        help="Directory to save extracted event clips (default: data/activitynet_clips)")
    parser.add_argument("--target-video-dir", default="data/activitynet_videos",
                        help="Directory containing original video files (.mp4) (default: data/activitynet_videos)")
    parser.add_argument("--log-file", default="data/logs/extract_clips_log.txt",
                        help="Log file path (default: data/logs/extract_clips_log.txt)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of worker threads (default: min(32, cpu_count()))")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create log directory if needed
    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = args.log_file
    
    # Initialize log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Event Clips Extraction Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    print(f"Starting event clips extraction...")
    print(f"Metadata: {args.metadata}")
    print(f"Event clip dir: {args.event_clip_dir}")
    print(f"Target video dir: {args.target_video_dir}")
    print(f"Log file: {log_file}")
    
    # Run main extraction
    extract_target_event_clips(
        metadata_path=args.metadata,
        event_clip_dir=args.event_clip_dir,
        target_video_dir=args.target_video_dir,
        log_file=log_file,
        max_workers=args.max_workers
    )
    
    # Write completion message to log
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Event Clips Extraction Log Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    print(f"Extraction completed. Log file saved to: {log_file}")
