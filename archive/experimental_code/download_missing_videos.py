#!/usr/bin/env python3
"""
Download missing validation videos from ManGO.
Only downloads videos that don't exist locally (no duplicates).

Usage:
    python download_missing_videos.py

Prerequisites:
    pip install python-irodsclient
"""

import os
import ssl
from pathlib import Path
import pandas as pd

try:
    from irods.session import iRODSSession
except ImportError:
    print("ERROR: python-irodsclient not installed")
    print("Install with: pip install python-irodsclient")
    exit(1)

from tqdm import tqdm


def main():
    # Paths
    epic_dir = Path(__file__).parent
    val_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_validation.csv"
    local_video_dir = epic_dir / "EPIC-KITCHENS" / "videos_640x360"

    # ManGO paths
    irods_base_path = "/set/home/ciis-lab/datasets/epic-kitchens/videos_640x360"

    # Load validation annotations
    df = pd.read_csv(val_csv)
    videos_needed = df[['participant_id', 'video_id']].drop_duplicates()

    print(f"Total validation segments: {len(df)}")
    print(f"Unique videos needed: {len(videos_needed)}")

    # Find missing videos
    missing_videos = []
    for _, row in videos_needed.iterrows():
        participant = row['participant_id']
        video_id = row['video_id']
        local_path = local_video_dir / participant / f"{video_id}.MP4"

        if not local_path.exists():
            missing_videos.append({
                'participant': participant,
                'video_id': video_id,
                'local_path': local_path,
                'irods_path': f"{irods_base_path}/{participant}/{video_id}.MP4"
            })

    print(f"Videos already downloaded: {len(videos_needed) - len(missing_videos)}")
    print(f"Videos to download: {len(missing_videos)}")

    if len(missing_videos) == 0:
        print("All videos already exist locally!")
        return

    # Connect to ManGO
    print("\nConnecting to ManGO...")
    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    ssl_settings = {"ssl_context": ssl_context}

    try:
        with iRODSSession(
            host="mango.kuleuven.be",
            port=1247,
            user=os.environ.get("IRODS_USER", input("ManGO username: ")),
            password=os.environ.get("IRODS_PASSWORD", input("ManGO password: ")),
            zone="set",
            **ssl_settings
        ) as session:
            print("Connected to ManGO!")

            # Download missing videos
            success = 0
            failed = []

            for video in tqdm(missing_videos, desc="Downloading"):
                try:
                    # Create local directory if needed
                    video['local_path'].parent.mkdir(parents=True, exist_ok=True)

                    # Check if file exists on ManGO
                    irods_path = video['irods_path']

                    try:
                        obj = session.data_objects.get(irods_path)

                        # Download file
                        local_path = str(video['local_path'])
                        with open(local_path, 'wb') as f:
                            with obj.open('r') as irods_f:
                                # Read in chunks
                                while True:
                                    chunk = irods_f.read(1024 * 1024)  # 1MB chunks
                                    if not chunk:
                                        break
                                    f.write(chunk)

                        success += 1

                    except Exception as e:
                        # File might not exist on ManGO
                        failed.append({
                            'video': f"{video['participant']}/{video['video_id']}.MP4",
                            'error': str(e)
                        })

                except Exception as e:
                    failed.append({
                        'video': f"{video['participant']}/{video['video_id']}.MP4",
                        'error': str(e)
                    })

            print(f"\nDownload complete!")
            print(f"  Successfully downloaded: {success}/{len(missing_videos)}")

            if failed:
                print(f"  Failed: {len(failed)}")
                for f in failed[:10]:
                    print(f"    - {f['video']}: {f['error'][:50]}")
                if len(failed) > 10:
                    print(f"    ... and {len(failed) - 10} more")

    except Exception as e:
        print(f"Failed to connect to ManGO: {e}")
        print("\nMake sure you have the correct credentials.")
        print("You can also set IRODS_USER and IRODS_PASSWORD environment variables.")


if __name__ == "__main__":
    main()
