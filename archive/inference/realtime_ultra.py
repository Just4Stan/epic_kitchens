"""
Ultra-fast EPIC-KITCHENS action recognition
Optimized for 30fps on M3 Pro

Press 'q' to quit
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque
import pandas as pd
from torchvision import transforms
import sys
from pathlib import Path
import time
import threading

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import ActionModel
from config import Config as cfg


class UltraFastRecognition:
    def __init__(self, checkpoint_path):
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Loading model...")
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

        # Half precision for speed
        self.model.half()
        print("Model loaded (FP16 mode)")

        # Class names
        epic_dir = Path(__file__).parent.parent
        verb_csv = pd.read_csv(epic_dir / cfg.VERB_CLASSES_CSV)
        noun_csv = pd.read_csv(epic_dir / cfg.NOUN_CLASSES_CSV)
        self.verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
        self.noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

        # Fast preprocessing - smaller size
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Frame buffer
        self.frame_buffer = deque(maxlen=90)  # 3 seconds

        # Async prediction
        self.current_pred = None
        self.pred_lock = threading.Lock()
        self.running = True

        # Time windows (frames at 30fps)
        self.windows = [30, 60, 90]  # 1s, 2s, 3s

    @torch.no_grad()
    def predict_async(self):
        """Background prediction thread."""
        while self.running:
            if len(self.frame_buffer) >= 30:
                best_pred = None
                best_conf = 0

                for window in self.windows:
                    if len(self.frame_buffer) >= window:
                        # Sample 16 frames
                        indices = np.linspace(
                            len(self.frame_buffer) - window,
                            len(self.frame_buffer) - 1,
                            16, dtype=int
                        )
                        frames = [self.frame_buffer[i] for i in indices]

                        x = torch.stack(frames).unsqueeze(0).to(self.device).half()

                        verb_logits, noun_logits = self.model(x)
                        verb_probs = F.softmax(verb_logits.float(), dim=1)[0]
                        noun_probs = F.softmax(noun_logits.float(), dim=1)[0]

                        v_conf, v_idx = verb_probs.max(0)
                        n_conf, n_idx = noun_probs.max(0)
                        avg = (v_conf.item() + n_conf.item()) / 2

                        if avg > best_conf:
                            best_conf = avg
                            best_pred = {
                                'verb': self.verb_map.get(v_idx.item(), "?"),
                                'noun': self.noun_map.get(n_idx.item(), "?"),
                                'conf': avg,
                                'window': f"{window//30}s"
                            }

                with self.pred_lock:
                    self.current_pred = best_pred

            time.sleep(0.1)  # Predict ~10 times per second

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Camera not found")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera: {w}x{h}")
        print("Press 'q' to quit\n")

        # Start background prediction thread
        pred_thread = threading.Thread(target=self.predict_async, daemon=True)
        pred_thread.start()

        frame_count = 0
        fps = 0
        fps_time = time.time()
        fps_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess and buffer (do this every frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.transform(rgb)
            self.frame_buffer.append(processed)

            # Get current prediction (thread-safe)
            with self.pred_lock:
                pred = self.current_pred

            # FPS calculation
            fps_frames += 1
            if fps_frames >= 15:
                fps = fps_frames / (time.time() - fps_time + 0.001)
                fps_time = time.time()
                fps_frames = 0

            # Simple overlay
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

            if pred:
                conf = pred['conf']
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255) if conf > 0.3 else (0, 0, 255)

                text = f"{pred['verb']} {pred['noun']}"
                cv2.putText(frame, text, (15, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"{conf*100:.0f}% [{pred['window']}]", (15, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                cv2.putText(frame, f"Buffering... {len(self.frame_buffer)}/30",
                           (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(frame, f"{fps:.0f} FPS", (w - 90, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow('EPIC-KITCHENS Ultra', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        self.running = False
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    epic_dir = Path(__file__).parent.parent
    if args.checkpoint is None:
        args.checkpoint = str(epic_dir / "outputs_exp15_h100/checkpoints/best_model.pth")

    if not Path(args.checkpoint).exists():
        print(f"ERROR: {args.checkpoint} not found")
        return

    UltraFastRecognition(args.checkpoint).run(args.camera)


if __name__ == '__main__':
    main()
