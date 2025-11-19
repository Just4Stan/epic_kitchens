# Quick script to add --dropout parameter to train_improved.py
import sys

with open('train_improved.py', 'r') as f:
    content = f.read()

# Add dropout parameter to argparser
if '--dropout' not in content:
    content = content.replace(
        "parser.add_argument('--lr', type=float, default=5e-5,\n                       help='Learning rate')",
        "parser.add_argument('--lr', type=float, default=5e-5,\n                       help='Learning rate')\n    parser.add_argument('--dropout', type=float, default=0.5,\n                       help='Dropout rate')"
    )
    
    # Modify model creation
    content = content.replace(
        "# Create model\n    model = get_improved_model(config).to(device)",
        "# Create model (with custom dropout if specified)\n    from model_improved import ImprovedActionRecognitionModel\n    dropout = args.dropout if hasattr(args, 'dropout') else 0.5\n    model = ImprovedActionRecognitionModel(\n        num_verb_classes=config.NUM_VERB_CLASSES,\n        num_noun_classes=config.NUM_NOUN_CLASSES,\n        num_frames=config.NUM_FRAMES,\n        dropout=dropout,\n        drop_path=0.1\n    ).to(device)"
    )
    
    with open('train_improved_v2.py', 'w') as f:
        f.write(content)
    print("Created train_improved_v2.py with dropout parameter")
else:
    print("Dropout parameter already exists")
