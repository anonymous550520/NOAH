"""
YouTube Video Downloader for ActivityNet Dataset
Downloads YouTube videos specified in a JSON file containing video IDs.
"""

import os
import sys
import json
import argparse
import logging
import glob
from pathlib import Path
import yt_dlp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_youtube_video(video_id, output_dir="data/activitynet_videos", quality="best"):
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, f'{video_id}.%(ext)s'),
            'format': quality,
            'noplaylist': True,
            'no_overwrites': True,
            'continue_dl': True,
            'ignoreerrors': False,
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return True
            
    except Exception as e:
        return False


def load_video_ids(json_file="data/ids.json"):
    try:
        with open(json_file, 'r') as f:
            video_ids = json.load(f)
        
        logging.info(f"Loaded {len(video_ids)} video IDs from {json_file}")
        return video_ids
        
    except FileNotFoundError:
        logging.error(f"File {json_file} not found")
        return []
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {json_file}")
        return []
    except Exception as e:
        logging.error(f"Error loading video IDs: {str(e)}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos from ids.json")
    parser.add_argument("--output", "-o", type=str, default="data/activitynet_videos",
                       help="Output directory (default: data/activitynet_videos)")
    parser.add_argument("--quality", "-q", type=str, default="best",
                       help="Video quality (default: best)")
    parser.add_argument("--ids-file", type=str, default="data/noah/ids.json",
                       help="Path to JSON file with video IDs (default: data/noah/ids.json)")
    parser.add_argument("--count", "-c", type=int, default=None, 
                       help="Number of videos to download (default: all)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel download threads (default: 4)")
    parser.add_argument("--log-file", default="data/logs/download_log.txt",
                       help="Log file path (default: data/logs/download_log.txt)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path(args.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Loading video IDs...")
    video_ids = load_video_ids(args.ids_file)
    if not video_ids:
        logging.error("No video IDs found. Exiting.")
        sys.exit(1)
    
    if args.count is not None:
        video_ids = video_ids[:args.count]
    
    logging.info(f"Will download {len(video_ids)} videos to {args.output}")
    logging.info(f"Using {args.max_workers} parallel workers...")
    
    successful = 0
    failed = 0
    skipped = 0
    
    def download_task(video_id):
        pattern = os.path.join(args.output, f'{video_id}.*')
        if glob.glob(pattern):
            return video_id, 'skipped'
        return video_id, 'success' if download_youtube_video(video_id, args.output, args.quality) else 'failed'
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_task, vid): vid for vid in video_ids}
        
        with tqdm(total=len(video_ids), desc="Downloading videos", unit="video") as pbar:
            for future in as_completed(futures):
                video_id, status = future.result()
                
                if status == 'skipped':
                    skipped += 1
                elif status == 'success':
                    successful += 1
                else:
                    failed += 1
                
                pbar.set_postfix({
                    'Success': successful,
                    'Failed': failed,
                    'Skipped': skipped,
                    'Rate': f"{successful}/{successful+failed}" if (successful+failed) > 0 else "0/0"
                })
                pbar.update(1)
    
    logging.info("Download complete!")
    logging.info(f"Total videos processed: {len(video_ids)}")
    logging.info(f"Successfully downloaded: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Skipped (already exist): {skipped}")
    if len(video_ids) > 0:
        logging.info(f"Success rate: {successful/len(video_ids)*100:.1f}%")


if __name__ == "__main__":
    main()
