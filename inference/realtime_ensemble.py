"""
Real-time EPIC-KITCHENS action recognition
Supports single model (fast) or ensemble mode

Press 'q' to quit, 's' to toggle single/ensemble, 'm' to cycle modes
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque
import pandas as pd
from torchvision import transforms
import argparse
import sys
from pathlib import Path
import threading
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import ActionModel
from config import Config as cfg


class RealtimeRecognition:
    def __init__(self, checkpoint_paths, weights=None, single_model=False):
        """
        Real-time action recognition from webcam.

        Args:
            checkpoint_paths: List of paths to model checkpoints
            weights: List of weights for each model (default: equal)
            single_model: If True, only use first model (faster)
        """
        self.single_model = single_model

        # Device selection - prefer MPS for M3 Pro
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load models
        self.models = []
        n_models = len(checkpoint_paths) if not single_model else 1

        if weights is None:
            self.weights = [1.0 / n_models] * n_models
        else:
            self.weights = weights[:n_models]
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        print(f"\nLoading {'1 model (FAST mode)' if single_model else f'{n_models} models (ensemble)'}...")
        for i, ckpt_path in enumerate(checkpoint_paths[:n_models]):
            print(f"  [{i+1}] {Path(ckpt_path).name}")
            model = self._load_model(ckpt_path)
            self.models.append(model)

        # Load class mappings
        self.load_class_mappings()

        # Frame preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Frame buffer - large enough for all window sizes
        self.frame_buffer = deque(maxlen=64)

        # Multi-scale predictions (short, medium, long action windows)
        self.scales = {
            'fast': 8,    # ~0.3s - quick actions
            'medium': 16,  # ~0.5s - normal actions
            'long': 32,   # ~1s - longer actions
        }
        self.current_mode = 'auto'  # 'auto', 'fast', 'medium', 'long'
        self.predictions = {k: None for k in self.scales}

        # For auto mode - pick highest confidence
        self.best_prediction = None

    def _load_model(self, checkpoint_path):
        """Load a single model from checkpoint."""
        model = ActionModel(
            num_verb_classes=cfg.NUM_VERB_CLASSES,
            num_noun_classes=cfg.NUM_NOUN_CLASSES,
            backbone=cfg.BACKBONE,
            temporal_model=cfg.TEMPORAL_MODEL,
            dropout=cfg.DROPOUT,
            num_frames=cfg.NUM_FRAMES
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def load_class_mappings(self):
        """Load verb and noun class names."""
        epic_dir = Path(__file__).parent.parent
        verb_csv = pd.read_csv(epic_dir / cfg.VERB_CLASSES_CSV)
        noun_csv = pd.read_csv(epic_dir / cfg.NOUN_CLASSES_CSV)

        self.verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
        self.noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

    def preprocess_frame(self, frame):
        """Preprocess a single frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)

    @torch.no_grad()
    def predict(self, num_frames):
        """Run prediction on buffered frames."""
        if len(self.frame_buffer) < num_frames:
            return None

        # Sample frames uniformly from buffer
        indices = np.linspace(0, len(self.frame_buffer) - 1, num_frames, dtype=int)
        sampled_frames = [self.frame_buffer[i] for i in indices]

        # Stack frames: (num_frames, 3, H, W)
        frames_tensor = torch.stack(sampled_frames, dim=0)

        # Add batch dimension: (1, num_frames, 3, H, W)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

        # Single model or ensemble
        if self.single_model or len(self.models) == 1:
            verb_logits, noun_logits = self.models[0](frames_tensor)
        else:
            # Ensemble - average logits
            verb_logits = torch.zeros(1, cfg.NUM_VERB_CLASSES, device=self.device)
            noun_logits = torch.zeros(1, cfg.NUM_NOUN_CLASSES, device=self.device)
            for model, weight in zip(self.models, self.weights):
                v, n = model(frames_tensor)
                verb_logits += weight * v
                noun_logits += weight * n

        # Convert to probabilities
        verb_probs = F.softmax(verb_logits, dim=1)[0]
        noun_probs = F.softmax(noun_logits, dim=1)[0]

        # Get top predictions
        verb_top_prob, verb_top_idx = verb_probs.topk(3)
        noun_top_prob, noun_top_idx = noun_probs.topk(3)

        return {
            'verb_name': self.verb_map.get(verb_top_idx[0].item(), "?"),
            'verb_conf': verb_top_prob[0].item(),
            'verb_top3': [(self.verb_map.get(idx.item(), "?"), prob.item())
                         for idx, prob in zip(verb_top_idx, verb_top_prob)],
            'noun_name': self.noun_map.get(noun_top_idx[0].item(), "?"),
            'noun_conf': noun_top_prob[0].item(),
            'noun_top3': [(self.noun_map.get(idx.item(), "?"), prob.item())
                         for idx, prob in zip(noun_top_idx, noun_top_prob)],
            'num_frames': num_frames,
        }

    def update_predictions(self):
        """Update predictions for all scales."""
        for scale_name, num_frames in self.scales.items():
            if len(self.frame_buffer) >= num_frames:
                self.predictions[scale_name] = self.predict(num_frames)

        # Auto mode: pick highest confidence prediction
        if self.current_mode == 'auto':
            best = None
            best_conf = 0
            for name, pred in self.predictions.items():
                if pred is not None:
                    conf = (pred['verb_conf'] + pred['noun_conf']) / 2
                    if conf > best_conf:
                        best_conf = conf
                        best = pred
                        best['scale'] = name
            self.best_prediction = best
        else:
            self.best_prediction = self.predictions.get(self.current_mode)
            if self.best_prediction:
                self.best_prediction['scale'] = self.current_mode

    def run(self, camera_id=0):
        """Run real-time webcam inference."""
        print(f"\nOpening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_id}")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera: {width}x{height}")
        print(f"\n{'='*50}")
        print("EPIC-KITCHENS Action Recognition")
        print(f"{'='*50}")
        print(f"Mode: {'SINGLE MODEL (fast)' if self.single_model else 'ENSEMBLE'}")
        print(f"\nControls:")
        print("  q - Quit")
        print("  s - Toggle single/ensemble mode")
        print("  m - Cycle scale: auto -> fast -> medium -> long")
        print(f"{'='*50}\n")

        frame_count = 0
        predict_every = 2 if self.single_model else 4  # More frequent for single model
        fps_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process and buffer frame
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer.append(processed_frame)

            # Run prediction periodically
            if frame_count % predict_every == 0 and len(self.frame_buffer) >= 8:
                self.update_predictions()

            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time + 0.001)
                fps_time = time.time()

            # Create display
            display = self.create_display(frame, width, height, fps)

            # Show
            cv2.imshow('EPIC-KITCHENS Action Recognition', display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Toggle single/ensemble
                self.single_model = not self.single_model
                predict_every = 2 if self.single_model else 4
                print(f"Mode: {'SINGLE' if self.single_model else 'ENSEMBLE'}")
            elif key == ord('m'):
                # Cycle modes
                modes = ['auto', 'fast', 'medium', 'long']
                idx = modes.index(self.current_mode)
                self.current_mode = modes[(idx + 1) % len(modes)]
                print(f"Scale: {self.current_mode}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def create_display(self, frame, width, height, fps):
        """Create the display overlay."""
        display = frame.copy()
        pred = self.best_prediction

        # Header overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        if pred is None:
            cv2.putText(display, f"Warming up... ({len(self.frame_buffer)}/8 frames)",
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Main action
            avg_conf = (pred['verb_conf'] + pred['noun_conf']) / 2
            color = self.conf_color(avg_conf)

            action = f"{pred['verb_name']} {pred['noun_name']}"
            cv2.putText(display, action, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

            # Confidence bar
            bar_w = int(avg_conf * 300)
            cv2.rectangle(display, (20, 65), (20 + bar_w, 85), color, -1)
            cv2.rectangle(display, (20, 65), (320, 85), (80, 80, 80), 2)
            cv2.putText(display, f"{avg_conf*100:.0f}%", (330, 82),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Scale indicator
            scale = pred.get('scale', self.current_mode)
            scale_colors = {'fast': (255, 200, 100), 'medium': (100, 255, 100), 'long': (100, 200, 255), 'auto': (255, 255, 255)}
            cv2.putText(display, f"[{scale.upper()}]", (420, 82),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, scale_colors.get(scale, (255,255,255)), 2)

            # Top alternatives
            y = 110
            cv2.putText(display, "Verbs:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1)
            cv2.putText(display, "Nouns:", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 150), 1)

            for i, ((v, vp), (n, np_)) in enumerate(zip(pred['verb_top3'], pred['noun_top3'])):
                cv2.putText(display, f"{v} {vp*100:.0f}%", (20, y + 18 + i*16),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1)
                cv2.putText(display, f"{n} {np_*100:.0f}%", (200, y + 18 + i*16),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1)

        # Status bar
        mode_text = "SINGLE" if self.single_model else "ENSEMBLE"
        cv2.putText(display, f"{mode_text} | {self.current_mode.upper()} | {fps:.0f} FPS",
                   (width - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Focus box
        if pred:
            avg_conf = (pred['verb_conf'] + pred['noun_conf']) / 2
            color = self.conf_color(avg_conf)
            box_size = int(min(width, height) * 0.4)
            cx, cy = width // 2, height // 2 + 40
            x1, y1 = cx - box_size // 2, cy - box_size // 2
            x2, y2 = cx + box_size // 2, cy + box_size // 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, max(2, int(avg_conf * 4)))

        # Controls hint
        cv2.putText(display, "q:quit s:single/ensemble m:scale", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return display

    def conf_color(self, conf):
        if conf > 0.5:
            return (0, 255, 0)
        elif conf > 0.25:
            return (0, 165, 255)
        return (0, 0, 255)


def main():
    parser = argparse.ArgumentParser(description='Real-time action recognition')
    parser.add_argument('--checkpoint1', type=str, default=None)
    parser.add_argument('--checkpoint2', type=str, default=None)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--single', action='store_true', help='Use single model (faster)')

    args = parser.parse_args()

    epic_dir = Path(__file__).parent.parent

    if args.checkpoint1 is None:
        args.checkpoint1 = str(epic_dir / "outputs_exp15_h100/checkpoints/best_model.pth")
    if args.checkpoint2 is None:
        args.checkpoint2 = str(epic_dir / "outputs/B2_cutmix_resume/checkpoints/best_model.pth")

    checkpoints = [args.checkpoint1]
    if not args.single and Path(args.checkpoint2).exists():
        checkpoints.append(args.checkpoint2)

    recognizer = RealtimeRecognition(
        checkpoint_paths=checkpoints,
        single_model=args.single or len(checkpoints) == 1
    )

    recognizer.run(camera_id=args.camera)


if __name__ == '__main__':
    main()
