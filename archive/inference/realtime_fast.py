"""
Fast real-time EPIC-KITCHENS action recognition (single model)
Multi-scale temporal windows: 1s, 2s, 3s, 4s - picks highest confidence

Press 'q' to quit
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
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import ActionModel
from config import Config as cfg


class FastActionRecognition:
    def __init__(self, checkpoint_path):
        # Device - MPS for M3
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Load single model
        print(f"Loading model: {Path(checkpoint_path).name}")
        self.model = ActionModel(
            num_verb_classes=cfg.NUM_VERB_CLASSES,
            num_noun_classes=cfg.NUM_NOUN_CLASSES,
            backbone=cfg.BACKBONE,
            temporal_model=cfg.TEMPORAL_MODEL,
            dropout=cfg.DROPOUT,
            num_frames=cfg.NUM_FRAMES
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model loaded!")

        # Load class names
        epic_dir = Path(__file__).parent.parent
        verb_csv = pd.read_csv(epic_dir / cfg.VERB_CLASSES_CSV)
        noun_csv = pd.read_csv(epic_dir / cfg.NOUN_CLASSES_CSV)
        self.verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
        self.noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Frame buffer - 4 seconds at 30fps = 120 frames max
        self.frame_buffer = deque(maxlen=120)

        # Time windows in frames (assuming ~30fps camera)
        # 1s=30, 2s=60, 3s=90, 4s=120
        self.windows = {
            '1s': 30,
            '2s': 60,
            '3s': 90,
            '4s': 120,
        }

    @torch.no_grad()
    def predict(self, num_buffer_frames):
        """Run prediction using num_buffer_frames from buffer, sampled to 16."""
        if len(self.frame_buffer) < num_buffer_frames:
            return None

        # Sample 16 frames uniformly from the window
        indices = np.linspace(
            len(self.frame_buffer) - num_buffer_frames,
            len(self.frame_buffer) - 1,
            16, dtype=int
        )
        frames = [self.frame_buffer[i] for i in indices]

        # Stack and add batch dim
        x = torch.stack(frames).unsqueeze(0).to(self.device)

        # Inference
        verb_logits, noun_logits = self.model(x)
        verb_probs = F.softmax(verb_logits, dim=1)[0]
        noun_probs = F.softmax(noun_logits, dim=1)[0]

        v_conf, v_idx = verb_probs.max(0)
        n_conf, n_idx = noun_probs.max(0)

        return {
            'verb': self.verb_map.get(v_idx.item(), "?"),
            'noun': self.noun_map.get(n_idx.item(), "?"),
            'verb_conf': v_conf.item(),
            'noun_conf': n_conf.item(),
            'avg_conf': (v_conf.item() + n_conf.item()) / 2,
        }

    def run(self, camera_id=0):
        print(f"\nOpening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera: {w}x{h}")
        print("\nRunning... Press 'q' to quit\n")

        frame_count = 0
        best_pred = None
        best_window = None
        fps = 0
        fps_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess and buffer
            processed = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.frame_buffer.append(processed)

            # Run multi-scale prediction every 5 frames
            if frame_count % 5 == 0 and len(self.frame_buffer) >= 30:
                best_conf = 0
                for name, window_frames in self.windows.items():
                    if len(self.frame_buffer) >= window_frames:
                        pred = self.predict(window_frames)
                        if pred and pred['avg_conf'] > best_conf:
                            best_conf = pred['avg_conf']
                            best_pred = pred
                            best_window = name

            # FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time + 0.001)
                fps_time = time.time()

            # Display
            display = frame.copy()

            # Dark header
            cv2.rectangle(display, (0, 0), (w, 120), (0, 0, 0), -1)

            if best_pred:
                conf = best_pred['avg_conf']
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255) if conf > 0.3 else (0, 0, 255)

                # Action text
                action = f"{best_pred['verb']} {best_pred['noun']}"
                cv2.putText(display, action, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

                # Confidence bar
                bar_w = int(conf * 400)
                cv2.rectangle(display, (20, 70), (20 + bar_w, 95), color, -1)
                cv2.rectangle(display, (20, 70), (420, 95), (80, 80, 80), 2)

                # Stats
                cv2.putText(display, f"{conf*100:.0f}%", (430, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"[{best_window}]", (500, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            else:
                cv2.putText(display, f"Buffering... {len(self.frame_buffer)}/30",
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # FPS counter
            cv2.putText(display, f"{fps:.0f} FPS", (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Focus box
            if best_pred:
                conf = best_pred['avg_conf']
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255) if conf > 0.3 else (0, 0, 255)
                box = int(min(w, h) * 0.35)
                cx, cy = w // 2, h // 2 + 30
                cv2.rectangle(display, (cx - box, cy - box), (cx + box, cy + box),
                             color, max(2, int(conf * 4)))

            cv2.imshow('EPIC-KITCHENS Fast', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    epic_dir = Path(__file__).parent.parent
    if args.checkpoint is None:
        # Use best single model (exp15)
        args.checkpoint = str(epic_dir / "outputs_exp15_h100/checkpoints/best_model.pth")

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    recognizer = FastActionRecognition(args.checkpoint)
    recognizer.run(camera_id=args.camera)


if __name__ == '__main__':
    main()
