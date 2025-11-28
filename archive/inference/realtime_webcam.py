"""
Real-time EPIC-KITCHENS action recognition with webcam
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


class RealtimeActionRecognition:
    def __init__(self, checkpoint_path, config, num_frames=8, fps=5):
        """
        Real-time action recognition from webcam.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Config object
            num_frames: Number of frames to use for prediction
            fps: How many times per second to make predictions
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

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = get_model(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from epoch {checkpoint['epoch']}")

        # Load class mappings
        self.load_class_mappings()

        # Frame preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Frame buffers - fast (8 frames) and slow (24 frames)
        self.frame_buffer_fast = deque(maxlen=8)
        self.frame_buffer_slow = deque(maxlen=24)

    def load_class_mappings(self):
        """Load verb and noun class names."""
        verb_csv = pd.read_csv(self.config.VERB_CLASSES_CSV)
        noun_csv = pd.read_csv(self.config.NOUN_CLASSES_CSV)

        self.verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
        self.noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

    def preprocess_frame(self, frame):
        """Preprocess a single frame."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)

    def predict(self, buffer, num_frames):
        """Run prediction on buffered frames."""
        if len(buffer) < num_frames:
            return None, None, None, None

        # Sample frames uniformly from buffer
        indices = np.linspace(0, len(buffer) - 1, num_frames, dtype=int)
        sampled_frames = [buffer[i] for i in indices]

        # Stack frames: (num_frames, 3, H, W)
        frames_tensor = torch.stack(sampled_frames, dim=0)

        # Add batch dimension: (1, num_frames, 3, H, W)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            verb_logits, noun_logits = self.model(frames_tensor)

            # Get probabilities
            verb_probs = torch.softmax(verb_logits, dim=1)[0]
            noun_probs = torch.softmax(noun_logits, dim=1)[0]

            # Get top predictions
            verb_top_prob, verb_top_idx = verb_probs.topk(1)
            noun_top_prob, noun_top_idx = noun_probs.topk(1)

            verb_name = self.verb_map.get(verb_top_idx[0].item(), "unknown")
            noun_name = self.noun_map.get(noun_top_idx[0].item(), "unknown")

            return verb_name, verb_top_prob[0].item(), noun_name, noun_top_prob[0].item()

    def run(self, camera_id=0):
        """Run real-time webcam inference."""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_id}")
            return

        # Get camera properties
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nCamera opened: {width}x{height} @ {cam_fps} FPS")
        print(f"\nDual Prediction Mode:")
        print(f"  FAST: 8 frames (~0.3s window), updates every frame")
        print(f"  SLOW: 24 frames (~0.8s window), updates every 10 frames")
        print(f"\nInstructions:")
        print(f"  - Point camera at kitchen actions (cutting, washing, opening, etc.)")
        print(f"  - Model was trained on egocentric POV footage")
        print(f"  - Press 'q' to quit\n")

        # Prediction state - FAST (short window)
        fast_verb = "warming up..."
        fast_noun = "warming up..."
        fast_verb_conf = 0.0
        fast_noun_conf = 0.0

        # Prediction state - SLOW (long window)
        slow_verb = "warming up..."
        slow_noun = "warming up..."
        slow_verb_conf = 0.0
        slow_noun_conf = 0.0

        frame_count = 0
        predict_fast_every = 1  # Fast prediction every frame
        predict_slow_every = 10  # Slow prediction every 10 frames

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Process and buffer frame to BOTH buffers
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer_fast.append(processed_frame)
            self.frame_buffer_slow.append(processed_frame)

            # FAST prediction (every frame)
            if frame_count % predict_fast_every == 0:
                result = self.predict(self.frame_buffer_fast, 8)
                if result[0] is not None:
                    fast_verb, fast_verb_conf, fast_noun, fast_noun_conf = result

            # SLOW prediction (every 10 frames)
            if frame_count % predict_slow_every == 0:
                result = self.predict(self.frame_buffer_slow, 8)
                if result[0] is not None:
                    slow_verb, slow_verb_conf, slow_noun, slow_noun_conf = result

            # Create display overlay
            display_frame = frame.copy()

            # Use SLOW prediction for main display (more stable)
            avg_conf = (slow_verb_conf + slow_noun_conf) / 2
            box_color = self.confidence_color(avg_conf)

            # Box size based on confidence
            box_width = int(width * 0.6)
            box_height = int(height * 0.6)
            center_x = width // 2
            center_y = height // 2

            x1 = center_x - box_width // 2
            y1 = center_y - box_height // 2
            x2 = center_x + box_width // 2
            y2 = center_y + box_height // 2

            # Draw center focus box with thickness based on confidence
            thickness = max(2, int(avg_conf * 6))
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, thickness)

            # Add corner markers
            corner_len = 30
            cv2.line(display_frame, (x1, y1), (x1 + corner_len, y1), box_color, thickness + 2)
            cv2.line(display_frame, (x1, y1), (x1, y1 + corner_len), box_color, thickness + 2)
            cv2.line(display_frame, (x2, y1), (x2 - corner_len, y1), box_color, thickness + 2)
            cv2.line(display_frame, (x2, y1), (x2, y1 + corner_len), box_color, thickness + 2)
            cv2.line(display_frame, (x1, y2), (x1 + corner_len, y2), box_color, thickness + 2)
            cv2.line(display_frame, (x1, y2), (x1, y2 - corner_len), box_color, thickness + 2)
            cv2.line(display_frame, (x2, y2), (x2 - corner_len, y2), box_color, thickness + 2)
            cv2.line(display_frame, (x2, y2), (x2, y2 - corner_len), box_color, thickness + 2)

            # Action label above box (SLOW prediction)
            action_text = f"SLOW: {slow_verb} {slow_noun}"
            text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = y1 - 20

            # Text background
            cv2.rectangle(display_frame,
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            cv2.putText(display_frame, action_text,
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            # Confidence percentage inside box
            conf_text = f"{avg_conf*100:.0f}%"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.putText(display_frame, conf_text,
                       (center_x - conf_size[0] // 2, center_y + conf_size[1] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color, 3)

            # Top panel - DUAL prediction display
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (width - 10, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

            # FAST prediction (top)
            cv2.putText(display_frame, "FAST (~0.3s):",
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            fast_avg = (fast_verb_conf + fast_noun_conf) / 2
            fast_color = self.confidence_color(fast_avg)
            cv2.putText(display_frame, f"{fast_verb} {fast_noun}",
                       (180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fast_color, 1)
            cv2.rectangle(display_frame, (180, 40), (180 + int(fast_avg * 250), 50), fast_color, -1)
            cv2.putText(display_frame, f"{fast_avg*100:.0f}%",
                       (440, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Separator
            cv2.line(display_frame, (20, 65), (width - 20, 65), (100, 100, 100), 1)

            # SLOW prediction (bottom)
            cv2.putText(display_frame, "SLOW (~0.8s):",
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            slow_avg = (slow_verb_conf + slow_noun_conf) / 2
            slow_color = self.confidence_color(slow_avg)
            cv2.putText(display_frame, f"{slow_verb} {slow_noun}",
                       (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, slow_color, 1)
            cv2.rectangle(display_frame, (180, 95), (180 + int(slow_avg * 250), 105), slow_color, -1)
            cv2.putText(display_frame, f"{slow_avg*100:.0f}%",
                       (440, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Buffer status
            cv2.putText(display_frame, f"Fast: {len(self.frame_buffer_fast)}/8",
                       (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
            cv2.putText(display_frame, f"Slow: {len(self.frame_buffer_slow)}/24",
                       (150, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

            # Show frame
            cv2.imshow('EPIC-KITCHENS Real-time Action Recognition', display_frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")

    def confidence_color(self, conf):
        """Get color based on confidence level."""
        if conf > 0.7:
            return (0, 255, 0)  # Green - high confidence
        elif conf > 0.4:
            return (0, 165, 255)  # Orange - medium
        else:
            return (0, 0, 255)  # Red - low confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time action recognition with webcam')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames for prediction (default: 8)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Predictions per second (default: 5)')

    args = parser.parse_args()

    config = Config()

    recognizer = RealtimeActionRecognition(
        args.checkpoint,
        config,
        num_frames=args.num_frames,
        fps=args.fps
    )

    recognizer.run(camera_id=args.camera)
