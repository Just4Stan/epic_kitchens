# GitHub Repository Setup - Using GitHub Desktop

## Your Repository is Ready!

**Location**: `/Users/stan/Downloads/RDLAB/epic_kitchens`
**Branch**: `main`
**Commits**: 2 commits ready to push
**Files**: 77 files (code only, no data/models)

---

## Step-by-Step Guide with GitHub Desktop

### Step 1: Open GitHub Desktop

1. Launch **GitHub Desktop** application
2. If not installed, download from: https://desktop.github.com/

### Step 2: Add Your Local Repository

1. In GitHub Desktop, click **File** → **Add Local Repository**
- Or use keyboard shortcut: `Cmd + O` (macOS)

2. Click **Choose...** and navigate to:
```
/Users/stan/Downloads/RDLAB/epic_kitchens
```

3. Click **Add Repository**

### Step 3: Publish to GitHub

1. You should see your repository with **2 commits**:
- `Refactor codebase to phase-based architecture`
- `Add GitHub repository setup instructions`

2. Click the **Publish repository** button (top right)

3. In the dialog that appears:
- **Name**: `epic-kitchens-action-recognition`
- **Description**: `Multi-phase action recognition on EPIC-KITCHENS-100 dataset`
- **Keep this code private**: Check this box (recommended)
- **Organization**: Leave as your personal account (or select team org if available)

4. Click **Publish Repository**

### Step 4: Verify Upload

GitHub Desktop will upload your code. You should see:
- Upload progress bar
- "Published" status when complete
- Button changes to "View on GitHub"

Click **View on GitHub** to open your repository in a browser.

---

## What Was Uploaded

### Included (77 files)
- All Python source code
- Documentation (README.md, ARCHITECTURE.md, etc.)
- Training scripts for all 3 phases
- SLURM job files
- Configuration files
- .gitignore

### Excluded (by .gitignore)
- Training data (EPIC-KITCHENS/, *.mp4, *.csv)
- Model checkpoints (*.pth, outputs_*)
- Training logs (training_*_output_*.txt)
- Virtual environments (venv/, epic_env/)
- Python cache files (__pycache__/)
- IDE files (.vscode/, .DS_Store)

**Repository size**: ~200 KB (code only)

---

## Share with Your Team

### Option 1: Add Collaborators Directly

1. On GitHub website, go to your repository
2. Click **Settings** tab
3. Click **Collaborators** (left sidebar)
4. Click **Add people**
5. Enter teammate's GitHub username or email
6. Select permission level:
- **Write**: Can push code
- **Admin**: Full access
7. Click **Add [username] to this repository**

### Option 2: Share Repository Link

Simply share this URL with your team:
```
https://github.com/YOUR_USERNAME/epic-kitchens-action-recognition
```

They can:
- **View** the code (if public)
- **Fork** it (create their own copy)
- **Clone** it (if you add them as collaborators)

---

## Team Members: How to Clone

### Using GitHub Desktop:

1. Open GitHub Desktop
2. Click **File** → **Clone Repository**
3. Search for: `epic-kitchens-action-recognition`
4. Choose local path
5. Click **Clone**

### Using Command Line:

```bash
git clone https://github.com/YOUR_USERNAME/epic-kitchens-action-recognition.git
cd epic-kitchens-action-recognition
```

---

## After Cloning - Setup Instructions for Team

Team members should follow these steps:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download EPIC-KITCHENS Dataset
- Visit: https://epic-kitchens.github.io/
- Download the dataset separately (NOT included in repo)
- Place in `EPIC-KITCHENS/` directory

### 3. Update Configuration
Edit `common/config.py` if paths are different:
```python
DATA_DIR = Path('EPIC-KITCHENS') # Update if needed
```

### 4. Start Training
Follow the guide in `docs/TRAINING_GUIDE.md`

---

## Making Changes and Syncing

### After You Make Code Changes:

1. **GitHub Desktop** will show changed files
2. Write a commit message (bottom left)
3. Click **Commit to main**
4. Click **Push origin** (top right) to upload changes

### To Get Team's Changes:

1. Click **Fetch origin** (top right)
2. If there are changes, click **Pull origin**

---

## Repository Statistics

| Metric | Value |
|--------|-------|
| Total Files | 77 |
| Total Lines | 11,254 |
| Repository Size | ~200 KB |
| Python Files | 76 |
| Phases | 3 (phase1, phase2, phase3) |
| Documentation Files | 7 |
| Excluded Data | ~100 GB+ |

---

## Important Notes

**Data Not Included**: Team members must download EPIC-KITCHENS-100 separately

**Models Not Included**: Trained checkpoints remain on:
- Your local machine: `/Users/stan/Downloads/RDLAB/epic_kitchens/outputs*/`
- VSC cluster: `/vsc-hard-mounts/leuven-data/380/vsc38064/epic_kitchens/outputs*/`

**Share Models Separately**: Use VSC, Google Drive, or other file sharing for checkpoints

---

## Troubleshooting

### "Repository already exists"
- Choose a different name or delete the existing one

### "Authentication failed"
- Click **GitHub Desktop** → **Preferences** → **Accounts**
- Sign in to GitHub

### "Large files detected"
- Double-check .gitignore is working
- GitHub Desktop should prevent committing large files

---

## Next Steps

1. Publish repository via GitHub Desktop
2. Verify on GitHub website
3. Add team collaborators
4. Share repository URL with team
5. Team clones and sets up their environment

**Last Updated**: November 19, 2025
