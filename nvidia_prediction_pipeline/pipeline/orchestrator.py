# pipeline/orchestrator.py


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from pipeline.train_all_models import train_all_models
from pipeline.train_meta_model import main as train_meta_model_main
from pipeline.predict_next_day import predict


def run_full_pipeline():
    print("\n[1] Training base models...")
    train_all_models()

    print("\n[2] Training meta-model (Ridge ensemble)...")
    train_meta_model_main()

    print("\n[3] Predicting next day's closing price with sentiment confidence...")
    predict()


if __name__ == "__main__":
    start = time.time()
    run_full_pipeline()
    end = time.time()
    print(f"\nPipeline completed in {end - start:.2f} seconds.")
