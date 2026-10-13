"""Run the whole classical-ML pipeline end to end (each step skips work that is already cached).

    python run_all.py            # everything
    python run_all.py 4 5 6      # only some steps
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    "1": "scripts/01_prepare_data.py",           # team split + Drive zips -> 256x256 images (~2 min)
    "2": "scripts/02_extract_features.py",       # ~6 min on 12 threads
    "3": "scripts/03_visualize_features.py",     # feature figures
    "4": "scripts/04_feature_combination_search.py",  # 63 combos x (SVM + LR), ~25 min
    "5": "scripts/05_train_final_models.py",     # full 11x11 grid on top-3 combos + test eval, ~50 min
    "6": "scripts/06_plot_results.py",           # evaluation figures
    "7": "scripts/07_build_writeup.py",          # Section 3 + 4.1 write-up (.docx/.md)
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STEPS)
    for k in wanted:
        print(f"\n=== step {k}: {STEPS[k]} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / STEPS[k])], check=True, cwd=ROOT)
    if "7" in wanted:
        print("\nSlides: node slides/build_slides.js")
