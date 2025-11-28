"""
Real-time EPIC-KITCHENS action recognition + YOLOv8 object detection
Combines action classification with object localization
Press 'q' to quit
"""

import torch
import cv2
import numpy as np
from collections import deque
import pandas as pd
from torchvision import transforms
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from common.config import Config
from phase1.model import get_model

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")


class RealtimeActionWithDetection:
    def __init__(self, checkpoint_path, config, num_frames=8, fps=5):
        """
        Real-time action recognition + object detection from webcam.

        Args:
            checkpoint_path: Path to EPIC-KITCHENS model checkpoint
            config: Config object
            num_frames: Number of frames for action prediction
            fps: How many times per second to make action predictions
        """
        self.config = config
        self.num_frames = num_frames
        self.fps = fps

        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using device: {self.device} (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load EPIC-KITCHENS action model
        print(f"\n[1/2] Loading EPIC-KITCHENS model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.action_model = get_model(config).to(self.device)
        self.action_model.load_state_dict(checkpoint['model_state_dict'])
        self.action_model.eval()
        print(f"✓ Action model loaded from epoch {checkpoint['epoch']}")

        # Load YOLOv8 object detection model
        if YOLO_AVAILABLE:
            print(f"\n[2/2] Loading YOLOv8 object detector...")
            self.yolo = YOLO('yolov8n.pt')  # Nano model (fastest)
            print(f"✓ YOLOv8 loaded")
        else:
            self.yolo = None
            print(f"✗ YOLOv8 not available - object detection disabled")

        # Load class mappings
        self.load_class_mappings()

        # Frame preprocessing for action model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Frame buffers - fast and slow
        self.frame_buffer_fast = deque(maxlen=8)
        self.frame_buffer_slow = deque(maxlen=24)

    def load_class_mappings(self):
        """Load verb and noun class names."""
        verb_csv = pd.read_csv(self.config.VERB_CLASSES_CSV)
        noun_csv = pd.read_csv(self.config.NOUN_CLASSES_CSV)

        self.verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
        self.noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

    def preprocess_frame(self, frame):
        """Preprocess a single frame for action model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)

    def predict_action(self, buffer, num_frames):
        """Run action prediction on buffered frames."""
        if len(buffer) < num_frames:
            return None, None, None, None

        # Sample frames uniformly
        indices = np.linspace(0, len(buffer) - 1, num_frames, dtype=int)
        sampled_frames = [buffer[i] for i in indices]

        # Stack and predict
        frames_tensor = torch.stack(sampled_frames, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            verb_logits, noun_logits = self.action_model(frames_tensor)

            verb_probs = torch.softmax(verb_logits, dim=1)[0]
            noun_probs = torch.softmax(noun_logits, dim=1)[0]

            verb_top_prob, verb_top_idx = verb_probs.topk(1)
            noun_top_prob, noun_top_idx = noun_probs.topk(1)

            verb_name = self.verb_map.get(verb_top_idx[0].item(), "unknown")
            noun_name = self.noun_map.get(noun_top_idx[0].item(), "unknown")

            return verb_name, verb_top_prob[0].item(), noun_name, noun_top_prob[0].item()

    def detect_objects(self, frame):
        """Run YOLOv8 object detection."""
        if self.yolo is None:
            return []

        results = self.yolo(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = results.names[cls]

            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class': class_name
            })

        return detections

    def confidence_color(self, conf):
        """Get color based on confidence level."""
        if conf > 0.7:
            return (0, 255, 0)  # Green
        elif conf > 0.4:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red

    def run(self, camera_id=0):
        """Run real-time inference with dual display."""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_id}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"\n{'='*60}")
        print(f"Camera: {width}x{height} @ {cam_fps:.1f} FPS")
        print(f"\nMODE: Action Recognition + Object Detection")
        print(f"  • ACTION: Fast (~0.3s) + Slow (~0.8s) predictions")
        print(f"  • OBJECTS: YOLOv8 real-time detection")
        print(f"\nControls: Press 'q' to quit")
        print(f"{'='*60}\n")

        # State
        fast_verb, fast_noun, fast_verb_conf, fast_noun_conf = "warming up...", "", 0.0, 0.0
        slow_verb, slow_noun, slow_verb_conf, slow_noun_conf = "warming up...", "", 0.0, 0.0

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Action recognition: buffer frames
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer_fast.append(processed_frame)
            self.frame_buffer_slow.append(processed_frame)

            # Fast action prediction (every frame)
            result = self.predict_action(self.frame_buffer_fast, 8)
            if result[0] is not None:
                fast_verb, fast_verb_conf, fast_noun, fast_noun_conf = result

            # Slow action prediction (every 10 frames)
            if frame_count % 10 == 0:
                result = self.predict_action(self.frame_buffer_slow, 8)
                if result[0] is not None:
                    slow_verb, slow_verb_conf, slow_noun, slow_noun_conf = result

            # Object detection (every frame)
            detections = self.detect_objects(frame)

            # Display
            display_frame = frame.copy()

            # Draw object detection bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                class_name = det['class']

                # Color by confidence
                color = self.confidence_color(conf)

                # Draw box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{class_name} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # Label background
                cv2.rectangle(display_frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 10, y1),
                            color, -1)
                cv2.putText(display_frame, label,
                           (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Top panel - Action predictions
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (width - 10, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

            # Title
            cv2.putText(display_frame, "ACTION RECOGNITION",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # FAST prediction
            cv2.putText(display_frame, "FAST:",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            fast_avg = (fast_verb_conf + fast_noun_conf) / 2
            fast_color = self.confidence_color(fast_avg)
            cv2.putText(display_frame, f"{fast_verb} {fast_noun}",
                       (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fast_color, 1)
            cv2.rectangle(display_frame, (100, 65), (100 + int(fast_avg * 200), 72), fast_color, -1)

            # SLOW prediction
            cv2.putText(display_frame, "SLOW:",
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            slow_avg = (slow_verb_conf + slow_noun_conf) / 2
            slow_color = self.confidence_color(slow_avg)
            cv2.putText(display_frame, f"{slow_verb} {slow_noun}",
                       (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, slow_color, 1)
            cv2.rectangle(display_frame, (100, 100), (100 + int(slow_avg * 200), 107), slow_color, -1)

            # Detection count
            cv2.putText(display_frame, f"Objects: {len(detections)}",
                       (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

            cv2.imshow('EPIC-KITCHENS + YOLOv8', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Action recognition + object detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to EPIC-KITCHENS model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames for action prediction (default: 8)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Action predictions per second (default: 5)')

    args = parser.parse_args()

    if not YOLO_AVAILABLE:
        print("\n" + "="*60)
        print("YOLOv8 not installed!")
        print("Install with: pip install ultralytics")
        print("="*60 + "\n")
        exit(1)

    config = Config()

    recognizer = RealtimeActionWithDetection(
        args.checkpoint,
        config,
        num_frames=args.num_frames,
        fps=args.fps
    )

    recognizer.run(camera_id=args.camera)
