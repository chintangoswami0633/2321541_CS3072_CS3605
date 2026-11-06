import argparse
import os
import sys
import time

try:
    import xgboost  # noqa: F401
except ImportError:
    print("Missing package: xgboost  ->  pip install xgboost")
    sys.exit(1)

try:
    import lightgbm  # noqa: F401
except ImportError:
    print("Missing package: lightgbm  ->  pip install lightgbm")
    sys.exit(1)

from gym_visualizer.config import ConfigManager
from gym_visualizer.data.generator import SyntheticDataGenerator
from gym_visualizer.data.loader import DataLoader
from gym_visualizer.evaluation.classification_evaluator import ClassificationEvaluator
from gym_visualizer.evaluation.feature_importance_extractor import (
    extract_classification_importances,
    extract_regression_importances,
    write_feature_importance_report,
)
from gym_visualizer.evaluation.image_pipeline_evaluator import ImagePipelineEvaluator
from gym_visualizer.evaluation.regression_evaluator import RegressionEvaluator
from gym_visualizer.evaluation.report_generator import ReportGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline.")
    parser.add_argument("--output-dir", default="evaluation_output/")
    args = parser.parse_args()
    out = args.output_dir

    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        print(f"Could not create output directory: {e}")
        return 1

    start = time.time()

    try:
        print("\n[Stage 1/6] Loading and preprocessing data...")
        config = ConfigManager()
        loader = DataLoader(config)
        try:
            ansur_df = loader.load_and_preprocess()
        except FileNotFoundError as e:
            print(f"ANSUR data files not found: {e}")
            return 1

        n_samples = config.get("training.synthetic_samples", 5000)
        synthetic_df = SyntheticDataGenerator(baseline_data=ansur_df, config=config).generate(n_samples=n_samples)

        print("\n[Stage 2/6] Running classification evaluation...")
        clf_eval = ClassificationEvaluator(data=synthetic_df, config=config, output_dir=out)
        clf_results = clf_eval.run()

        print("\n[Stage 3/6] Running regression evaluation...")
        reg_eval = RegressionEvaluator(data=synthetic_df, config=config, output_dir=out)
        single_results = reg_eval.run_single()
        multi_results  = reg_eval.run_multi()

        print("\n[Stage 4/6] Extracting feature importances...")
        clf_imp_paths, clf_imp_dfs = extract_classification_importances(clf_eval, out)
        reg_imp_paths, reg_imp_dfs = extract_regression_importances(reg_eval, out)
        fi_report = write_feature_importance_report(clf_imp_dfs + reg_imp_dfs, out)

        print("\n[Stage 5/6] Running image pipeline evaluation...")
        img_eval = ImagePipelineEvaluator(cache_dir="cache/images/", output_dir=out)
        img_results = img_eval.run()

        print("\n[Stage 6/6] Generating summary report...")
        artefacts = (
            clf_eval.artefact_paths
            + reg_eval.artefact_paths
            + clf_imp_paths
            + reg_imp_paths
            + [fi_report]
            + img_eval.artefact_paths
        )
        ReportGenerator(out).generate(
            classification_results=clf_results,
            single_regression_results=single_results,
            multi_regression_results=multi_results,
            image_results=img_results,
            artefact_paths=artefacts,
            elapsed_seconds=time.time() - start,
        )

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
