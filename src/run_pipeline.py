"""
FactoryGuard AI - Pipeline Runner
End-to-end ML pipeline orchestration: ingest → preprocess → engineer → tune → train → evaluate → explain.
"""

import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def run_full_pipeline(skip_tuning=False):
    """Run the complete ML pipeline end-to-end."""
    total_start = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   FactoryGuard AI – Predictive Maintenance Pipeline      ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")

    # ── Stage 1: Data Ingestion ──────────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 1: Data Ingestion")
    from src.data_ingestion import run_ingestion
    df_cleaned = run_ingestion()
    logger.info(f"  ✓ Stage 1 completed in {time.time() - stage_start:.1f}s\n")

    # ── Stage 2: Preprocessing ───────────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 2: Preprocessing")
    from src.preprocessing import run_preprocessing
    df_preprocessed = run_preprocessing()
    logger.info(f"  ✓ Stage 2 completed in {time.time() - stage_start:.1f}s\n")

    # ── Stage 3: Feature Engineering ─────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 3: Feature Engineering")
    from src.feature_engineering import run_feature_engineering
    df_features = run_feature_engineering(df_preprocessed)
    logger.info(f"  ✓ Stage 3 completed in {time.time() - stage_start:.1f}s\n")

    # ── Stage 3.5: Hyperparameter Tuning (Optional) ─────────────
    tuned_params = None
    if not skip_tuning:
        stage_start = time.time()
        logger.info("▶ Starting Stage 3.5: Hyperparameter Tuning")
        from src.hyperparameter_tuning import run_tuning
        tuned_params = run_tuning()
        logger.info(f"  ✓ Stage 3.5 completed in {time.time() - stage_start:.1f}s\n")
    else:
        logger.info("⏭ Skipping hyperparameter tuning (--skip-tuning flag)\n")

    # ── Stage 4: Model Training ──────────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 4: Model Training")
    from src.model_training import run_training
    best_model, scaler, X_train, X_test, y_train, y_test, feature_cols, all_results = run_training(tuned_params)
    logger.info(f"  ✓ Stage 4 completed in {time.time() - stage_start:.1f}s\n")

    # ── Stage 5: Model Evaluation ────────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 5: Model Evaluation")
    from src.model_evaluation import evaluate_model
    import json
    from src.config import BEST_PARAMS_PATH

    with open(BEST_PARAMS_PATH, "r") as f:
        metadata = json.load(f)
    model_name = metadata.get("best_model_name", "Best Model")

    eval_results = evaluate_model(best_model, X_test, y_test, model_name)
    logger.info(f"  ✓ Stage 5 completed in {time.time() - stage_start:.1f}s\n")

    # ── Stage 6: Explainability ──────────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 6: SHAP Explainability")
    from src.explainability import run_explainability
    explainer, shap_values, importance_df = run_explainability(
        best_model, X_test, feature_cols
    )
    logger.info(f"  ✓ Stage 6 completed in {time.time() - stage_start:.1f}s\n")

    # ── Stage 7: Sample Prediction ───────────────────────────────
    stage_start = time.time()
    logger.info("▶ Starting Stage 7: Sample Prediction")
    from src.predict import run_sample_prediction
    run_sample_prediction()
    logger.info(f"  ✓ Stage 7 completed in {time.time() - stage_start:.1f}s\n")

    # ── Summary ──────────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║              PIPELINE COMPLETE                           ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Best model: {model_name}")
    logger.info(f"PR-AUC: {eval_results['pr_auc']:.4f}")
    logger.info(f"F1 Score: {eval_results['f1']:.4f}")
    logger.info(f"Precision: {eval_results['precision']:.4f}")
    logger.info(f"Recall: {eval_results['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="FactoryGuard AI Pipeline Runner")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning (uses default params)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["ingest", "preprocess", "features", "tune", "train", "evaluate", "explain", "predict"],
        help="Run a single pipeline stage",
    )

    args = parser.parse_args()

    if args.stage:
        run_single_stage(args.stage)
    else:
        run_full_pipeline(skip_tuning=args.skip_tuning)


def run_single_stage(stage):
    """Run a single pipeline stage."""
    logger.info(f"Running single stage: {stage}")

    if stage == "ingest":
        from src.data_ingestion import run_ingestion
        run_ingestion()
    elif stage == "preprocess":
        from src.preprocessing import run_preprocessing
        run_preprocessing()
    elif stage == "features":
        from src.preprocessing import run_preprocessing
        from src.feature_engineering import run_feature_engineering
        df = run_preprocessing()
        run_feature_engineering(df)
    elif stage == "tune":
        from src.hyperparameter_tuning import run_tuning
        run_tuning()
    elif stage == "train":
        from src.model_training import run_training
        run_training()
    elif stage == "evaluate":
        from src.model_evaluation import main as eval_main
        eval_main()
    elif stage == "explain":
        from src.explainability import main as explain_main
        explain_main()
    elif stage == "predict":
        from src.predict import run_sample_prediction
        run_sample_prediction()


if __name__ == "__main__":
    main()