"""
List available cameras on macOS
Run from epic_kitchens/ directory: python inference/list_cameras.py
"""

import cv2

print("Detecting available cameras...\n")

available_cameras = []

# Check first 10 camera indices
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        # Get camera info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera {i}:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")

        # Try to read a frame to verify it works
        ret, frame = cap.read()
        if ret:
            print(f"  Status: ✓ Working")
            available_cameras.append(i)
        else:
            print(f"  Status: ✗ Can't read frames")

        print()
        cap.release()

if available_cameras:
    print(f"\nFound {len(available_cameras)} working camera(s): {available_cameras}")
    print(f"\nTo use a camera, run:")
    print(f"  python realtime_webcam.py --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth --camera <ID>")

    if len(available_cameras) > 1:
        print(f"\nYou have multiple cameras:")
        print(f"  Camera 0 is typically your built-in FaceTime camera")
        print(f"  Camera 1+ could be your iPhone via Continuity Camera")
else:
    print("No cameras found!")
    print("\nFor iPhone Continuity Camera:")
    print("  1. Make sure iPhone and Mac are on same Apple ID")
    print("  2. Both have WiFi and Bluetooth enabled")
    print("  3. iPhone is unlocked and nearby")
    print("  4. macOS Ventura 13.0+ and iOS 16+")
