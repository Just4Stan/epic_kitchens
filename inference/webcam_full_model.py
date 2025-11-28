#!/usr/bin/env python3
"""
Real-time webcam action recognition using full EPIC-KITCHENS model.
Model architecture matches train_full.py from VSC training.

Usage:
    python inference/webcam_full_model.py --checkpoint outputs/full_32frames_v1/checkpoints/best_model.pth

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
    def __init__(self, checkpoint_path, num_frames=16):
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using: Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using: CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print(f"Using: CPU")

        self.num_frames = num_frames

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
        print(f"Model loaded! ({num_frames} frames)")

        # Load class names
        epic_dir = Path(__file__).parent.parent
        verb_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_verb_classes.csv"
        noun_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_noun_classes.csv"

        verb_df = pd.read_csv(verb_csv)
        noun_df = pd.read_csv(noun_csv)
        self.verb_map = dict(zip(verb_df['id'], verb_df['key']))
        self.noun_map = dict(zip(noun_df['id'], noun_df['key']))

        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Frame buffer
        self.frame_buffer = deque(maxlen=num_frames * 2)  # Keep extra frames

    def preprocess(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)

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

        verb_probs = torch.softmax(verb_logits, dim=1)[0]
        noun_probs = torch.softmax(noun_logits, dim=1)[0]

        verb_conf, verb_idx = verb_probs.max(0)
        noun_conf, noun_idx = noun_probs.max(0)

        return {
            'verb': self.verb_map.get(verb_idx.item(), '?'),
            'verb_conf': verb_conf.item(),
            'noun': self.noun_map.get(noun_idx.item(), '?'),
            'noun_conf': noun_conf.item()
        }

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_id}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {width}x{height}")
        print(f"\nPress 'q' to quit\n")

        current_pred = None
        frame_count = 0
        predict_every = 3  # Predict every N frames for speed

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Buffer frame
            self.frame_buffer.append(self.preprocess(frame))

            # Predict periodically
            if frame_count % predict_every == 0:
                pred = self.predict()
                if pred:
                    current_pred = pred

            # Display
            display = frame.copy()

            # Dark overlay at top
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            # Title
            cv2.putText(display, "EPIC-KITCHENS Action Recognition", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Model: 32-frame ResNet50+LSTM (35% accuracy)", (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            if current_pred:
                avg_conf = (current_pred['verb_conf'] + current_pred['noun_conf']) / 2

                # Color based on confidence
                if avg_conf > 0.6:
                    color = (0, 255, 0)  # Green
                elif avg_conf > 0.3:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 100, 255)  # Red

                # Action prediction
                action_text = f"{current_pred['verb']} {current_pred['noun']}"
                cv2.putText(display, action_text, (20, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                # Confidence bar
                bar_width = int(avg_conf * 200)
                cv2.rectangle(display, (20, 105), (20 + bar_width, 115), color, -1)
                cv2.rectangle(display, (20, 105), (220, 115), (100, 100, 100), 1)
                cv2.putText(display, f"{avg_conf*100:.0f}%", (230, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Center focus box
                box_size = int(min(width, height) * 0.5)
                cx, cy = width // 2, height // 2
                x1, y1 = cx - box_size // 2, cy - box_size // 2
                x2, y2 = cx + box_size // 2, cy + box_size // 2

                thickness = max(2, int(avg_conf * 4))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

                # Confidence in center
                conf_text = f"{avg_conf*100:.0f}%"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                cv2.putText(display, conf_text,
                           (cx - text_size[0] // 2, cy + text_size[1] // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            else:
                cv2.putText(display, f"Buffering... {len(self.frame_buffer)}/{self.num_frames}",
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            cv2.imshow('Action Recognition', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames (use 16 for 16-frame model, 32 for 32-frame)')
    args = parser.parse_args()

    recognizer = WebcamActionRecognition(args.checkpoint, num_frames=args.num_frames)
    recognizer.run(camera_id=args.camera)
