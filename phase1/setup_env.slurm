#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=setup_epic_env
#SBATCH --output=setup_output_%j.txt
#SBATCH --error=setup_error_%j.txt

echo "Setting up Python environment for EPIC-KITCHENS..."
echo "Start time: $(date)"
echo ""

module purge
module load cluster/wice/interactive
module load Python/3.11.3-GCCcore-12.3.0

cd $VSC_DATA/epic_kitchens

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv epic_env
source epic_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing PyTorch and dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing other packages..."
pip install pandas numpy opencv-python tqdm

echo ""
echo "Installed packages:"
pip list

echo ""
echo "Setup complete!"
echo "End time: $(date)"
