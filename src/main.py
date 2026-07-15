"""
Main Entry Point - AI-Driven Occupancy Detection
=================================================
Orchestrates the complete pipeline:
1. Data Loading & Feature Engineering
2. EDA Visualizations
3. Model Training with GridSearchCV
4. Model Comparison Visualizations

Usage:
    python src/main.py
    python src/main.py --data path/to/Occupancy_Estimation.csv
    python src/main.py --output results/
"""

import argparse
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import process_data, load_data
from src.model_training import train_with_grid_search
from src.visualization import run_eda_visualizations, run_model_visualizations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Driven Occupancy Detection Pipeline"
    )
    parser.add_argument(
        "--data", type=str, default="data/Occupancy_Estimation.csv",
        help="Path to the CSV file (default: data/Occupancy_Estimation.csv)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Cross-validation folds for GridSearchCV (default: 5)",
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA visualizations",
    )
    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_args()
    
    # Find data file
    if not os.path.exists(args.data):
        alt_paths = [
            "data/Occupancy_Estimation.csv",
            "Occupancy_Estimation.csv",
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                args.data = p
                found = True
                break
        if not found:
            print(f"❌ Data file not found: '{args.data}'")
            print("   Place 'Occupancy_Estimation.csv' in data/ directory")
            sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    print("╔" + "═" * 58 + "╗")
    print("║  AI-DRIVEN OCCUPANCY DETECTION - ML PIPELINE              ║")
    print("╚" + "═" * 58 + "╝")
    
    # EDA on raw data
    if not args.skip_eda:
        raw_df = load_data(args.data)
        run_eda_visualizations(raw_df, args.output)
    
    # Process data
    X_train, X_test, y_train, y_test, scaler, feature_names, raw_df = process_data(
        args.data
    )
    
    # Train models
    results_df, detailed_results = train_with_grid_search(
        X_train, y_train, X_test, y_test, cv=args.cv
    )
    
    # Model visualizations
    run_model_visualizations(results_df, detailed_results, args.output)
    
    # Save results
    results_path = os.path.join(args.output, "model_results.csv")
    results_df.to_csv(results_path)
    print(f"\n📁 Results saved: {results_path}")
    
    print("\n╔" + "═" * 58 + "╗")
    print("║                    PIPELINE COMPLETE!                     ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n📂 All outputs in: {args.output}/")


if __name__ == "__main__":
    main()
