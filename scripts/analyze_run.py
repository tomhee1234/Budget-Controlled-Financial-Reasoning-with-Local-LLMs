"""Analyze a new run in its own directory, keeping other results untouched."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
import analyze_results
import compare_conditions
import study_analysis
paths = sorted(args.run_dir.glob('*/experiment.csv'))
analyze_results.OUTPUT_DIR = str(args.run_dir / 'analysis')
compare_conditions.OUTPUT_PATH = str(args.run_dir / 'analysis/statistical_tests.md')
analyze_results.main(paths)
compare_conditions.main(paths)
study_analysis.analyze(paths, args.run_dir / 'analysis')
