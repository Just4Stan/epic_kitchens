#!/usr/bin/env python3
"""
Real-time webcam action recognition using full EPIC-KITCHENS model.
Model architecture matches train_full.py from VSC training.

OPTIMIZATIONS:
- FP16 inference on Apple Silicon (2x faster)
- Frame skipping for configurable FPS
- Torch compile for JIT optimization
- FPS counter

Usage:
    python inference/webcam_full_model.py --checkpoint outputs/full_a100_v3/checkpoints/best_model.pth --num_frames 16 --fp16

Press 'q' to quit
"""

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from collections import deque
import pandas as pd
from torchvision import transforms
import argparse
from pathlib import Path
import time


class ActionModel(nn.Module):
    """Action recognition model - matches train_full.py architecture."""

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 backbone='resnet50', temporal_model='lstm',
                 dropout=0.5, num_frames=16):
        super().__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
            self.backbone.fc = nn.Identity()

        self.temporal_model = temporal_model
        if temporal_model == 'lstm':
            self.temporal = nn.LSTM(
                self.feature_dim, self.feature_dim // 2,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.temporal_dim = self.feature_dim
        elif temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.temporal_dim = self.feature_dim
        else:
            self.temporal = None
            self.temporal_dim = self.feature_dim

        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(self.temporal_dim, num_verb_classes)
        self.noun_head = nn.Linear(self.temporal_dim, num_noun_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)

        if self.temporal_model == 'lstm':
            temporal_out, _ = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        elif self.temporal_model == 'transformer':
            temporal_out = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        else:
            pooled = features.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.verb_head(pooled), self.noun_head(pooled)


class WebcamActionRecognition:
    def __init__(self, checkpoint_path, num_frames=16, use_fp16=False, skip_frames=2):
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"🚀 Using: Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 Using: CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print(f"⚠️  Using: CPU")

        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.use_fp16 = use_fp16 and self.device.type == 'mps'

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = ActionModel(
            num_verb_classes=97,
            num_noun_classes=300,
            backbone='resnet50',
            temporal_model='lstm',
            dropout=0.0,  # No dropout during inference
            num_frames=num_frames
        ).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # FP16 optimization for Apple Silicon
        if self.use_fp16:
            self.model = self.model.half()
            print(f"✅ FP16 enabled (2x faster)")

        # Try torch.compile for JIT optimization
        try:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            print(f"✅ Model compiled with torch.compile")
        except:
            pass

        print(f"Model loaded! ({num_frames} frames, skip={skip_frames})")

        # Load class names
        epic_dir = Path(__file__).parent.parent
        verb_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_verb_classes.csv"
        noun_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_noun_classes.csv"

        verb_df = pd.read_csv(verb_csv)
        noun_df = pd.read_csv(noun_csv)
        self.verb_map = dict(zip(verb_df['id'], verb_df['key']))
        self.noun_map = dict(zip(noun_df['id'], noun_df['key']))

        # Transform (optimized)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Frame buffer (compact)
        self.frame_buffer = deque(maxlen=num_frames)

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_inference_time = time.time()

    def preprocess(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb)
        if self.use_fp16:
            tensor = tensor.half()
        return tensor

    @torch.no_grad()
    def predict(self):
        if len(self.frame_buffer) < self.num_frames:
            return None

        # Sample num_frames uniformly from buffer
        indices = np.linspace(0, len(self.frame_buffer) - 1, self.num_frames, dtype=int)
        frames = [self.frame_buffer[i] for i in indices]

        # Stack: (1, T, C, H, W)
        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)

        verb_logits, noun_logits = self.model(frames_tensor)

        # Convert to FP32 for softmax if using FP16
        if self.use_fp16:
            verb_logits = verb_logits.float()
            noun_logits = noun_logits.float()

        verb_probs = torch.softmax(verb_logits, dim=1)[0]
        noun_probs = torch.softmax(noun_logits, dim=1)[0]

        verb_conf, verb_idx = verb_probs.max(0)
        noun_conf, noun_idx = noun_probs.max(0)

        # Track FPS
        current_time = time.time()
        elapsed = current_time - self.last_inference_time
        if elapsed > 0:
            self.fps_history.append(1.0 / elapsed)
        self.last_inference_time = current_time

        return {
            'verb': self.verb_map.get(verb_idx.item(), '?'),
            'verb_conf': verb_conf.item(),
            'noun': self.noun_map.get(noun_idx.item(), '?'),
            'noun_conf': noun_conf.item()
        }

    def run(self, camera_id=0, show_fps=True):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open camera {camera_id}")
            return

        # Optimize camera settings for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Camera: {width}x{height}")
        print(f"⚡ Processing every {self.skip_frames} frames")
        print(f"🎯 Buffer: {self.num_frames} frames")
        print(f"\nPress 'q' to quit\n")

        current_pred = None
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Buffer frame
            self.frame_buffer.append(self.preprocess(frame))

            # Predict every skip_frames
            if frame_count % self.skip_frames == 0:
                pred = self.predict()
                if pred:
                    current_pred = pred

            # Display (optimized)
            display = cv2.resize(frame, (width, height))

            # FPS counter
            if show_fps and len(self.fps_history) > 0:
                avg_fps = np.mean(list(self.fps_history))
                cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if current_pred:
                avg_conf = (current_pred['verb_conf'] + current_pred['noun_conf']) / 2

                # Color based on confidence
                color = (0, 255, 0) if avg_conf > 0.5 else (0, 165, 255)

                # Action prediction (simplified for speed)
                action_text = f"{current_pred['verb']} {current_pred['noun']}"
                cv2.putText(display, action_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"{avg_conf*100:.0f}%", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(display, f"Buffering... {len(self.frame_buffer)}/{self.num_frames}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            cv2.imshow('Action Recognition (Optimized)', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print("PERFORMANCE STATISTICS")
        print(f"{'='*50}")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        if len(self.fps_history) > 0:
            print(f"Inference FPS: {np.mean(list(self.fps_history)):.1f}")
            print(f"Peak FPS: {max(self.fps_history):.1f}")
        print(f"{'='*50}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized webcam action recognition')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames (16 fastest, 32 most accurate)')
    parser.add_argument('--skip_frames', type=int, default=2,
                       help='Process every Nth frame (lower=more accurate, higher=faster)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 for 2x speedup on Apple Silicon')
    parser.add_argument('--no_fps', action='store_true',
                       help='Hide FPS counter')
    args = parser.parse_args()

    recognizer = WebcamActionRecognition(
        args.checkpoint,
        num_frames=args.num_frames,
        use_fp16=args.fp16,
        skip_frames=args.skip_frames
    )
    recognizer.run(camera_id=args.camera, show_fps=not args.no_fps)
