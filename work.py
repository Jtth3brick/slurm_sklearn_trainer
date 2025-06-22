import argparse
import logging
import pickle
import random
import signal
import time
from typing import Dict, Optional

import pandas as pd
import redis
from sklearn.metrics import roc_auc_score

from utils import SPLIT_DIR, ModelConfig, SplitConfig

# note: SIGALRM won't work on non-unix systems, namely windows
MAX_FIT_TIME = 60 * 45  # 45 minutes
MAX_UNCAUGHT_FAILURES = 3


def fitting_handler(signum, frame):
    raise TimeoutError("Model failed to train in time.")


def get_arg(r, num_attempts=3) -> Optional[ModelConfig]:
    for attempt in range(num_attempts):
        try:
            item = r.brpop("model_queue", timeout=1)
            if not item:
                logging.info("No items in queue")
                return None

            model_config: ModelConfig = pickle.loads(item[1])
            return model_config

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {e}")
            if attempt < num_attempts - 1:
                sleep_time = random.uniform(0.5, 2.0)
                logging.info(f"Sleeping {sleep_time:.2f} seconds before retry")
                time.sleep(sleep_time)
            else:
                logging.error(f"All {num_attempts} attempts failed")

    return None


def put_result(r, result: ModelConfig):
    """Store a completed result in Redis with hash as key"""
    pickled_result = pickle.dumps(result)

    # Store the result with hash as key
    r.set(f"result:{result.config_hash}", pickled_result)

    # Add to set of all results for easy listing/counting
    r.sadd("results", result.config_hash)

    # Log progress
    queue_remaining = r.llen("model_queue")
    completed_count = r.scard("results")

    logging.info(f"Stored result with hash: {result.config_hash}")
    logging.info(
        f"Progress: {completed_count} completed, {queue_remaining} remaining in queue"
    )


def ensure_split(
    split_cache: Dict[str, SplitConfig], split_id: str, max_size: Optional[int] = None
):
    """Load the split config if we don't already have it"""
    if split_id in split_cache:
        return

    split_path = f"{SPLIT_DIR}/split_{split_id}.pkl"
    with open(split_path, "rb") as f:
        split_config: SplitConfig = pickle.load(f)
    split_cache[split_id] = split_config

    while max_size and len(split_cache) > max_size:
        # Remove first key that isn't our current split_id
        for key in split_cache:
            if key != split_id:
                del split_cache[key]
                break


def fit(model_config: ModelConfig, split_config: SplitConfig) -> ModelConfig:
    """Processes ModelConfig including CV scores if requested. Scoring metric is AUC"""

    assert model_config.split_id == split_config.split_id, (
        "Data does not match. Exiting..."
    )

    # Get cv scores if requested
    if split_config.cv_indices:
        for i, (train_idx, val_idx) in enumerate(split_config.cv_indices):
            logging.info(f"Training cv {i} of {len(split_config.cv_indices)}")

            assert split_config.X_cv is not None and isinstance(
                split_config.X_cv, pd.DataFrame
            )
            assert split_config.y_cv is not None and isinstance(
                split_config.y_cv, pd.Series
            )

            X_train_fold = split_config.X_cv.loc[train_idx].copy()
            y_train_fold = split_config.y_cv.loc[train_idx].copy()
            X_val_fold = split_config.X_cv.loc[val_idx]
            y_val_fold = split_config.y_cv.loc[val_idx]

            # Fit the model
            pipe = model_config.get_empty_pipe()
            pipe.fit(X_train_fold, y_train_fold)

            # Add cv score to result
            pred_proba = pipe.predict_proba(X_val_fold)[:, 1]
            model_config.cv_scores.append(float(roc_auc_score(y_val_fold, pred_proba)))
    else:
        logging.info("Skipping CV fitting")

    # Fit traditional train/validate if requested
    if split_config.X_train is not None:
        assert split_config.y_train is not None
        logging.info("Fitting full model")
        pipe = model_config.get_empty_pipe()
        pipe.fit(split_config.X_train.copy(), split_config.y_train.copy())
        pred_proba = pipe.predict_proba(split_config.X_val)[:, 1]
        model_config.validate_score = float(
            roc_auc_score(split_config.y_val, pred_proba)
        )
    else:
        logging.info("Skipping train/val fitting")

    return model_config


def main(worker_id: int):
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(asctime)s - worker-{worker_id} - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info(f"worker active")

    with open("redis_connection.txt", "r") as f:
        host, port = f.read().strip().split(":")

    r = redis.Redis(host=host, port=int(port), decode_responses=False)
    logging.info(f"worker successfuly connected to redis")

    # Used to store datasets without keeping too many in memory
    split_cache: Dict[str, SplitConfig] = {}

    # Begin training loop with timeout
    trained_model_count = 0
    uncaught_failures = 0
    while True:
        try:
            model_config = get_arg(r, num_attempts=3)
            if model_config is None:
                logging.info("Could not get arg. Queue is likely empty.")
                break

            # Load split and only keep one in memory at a time
            # (Queue ordering from manager prevents unnecessary misses)
            ensure_split(split_cache, split_id=model_config.split_id, max_size=1)

            try:
                logging.info(f"fitting for {model_config} starting")
                signal.signal(signal.SIGALRM, fitting_handler)
                signal.alarm(MAX_FIT_TIME)
                fit(
                    model_config=model_config,
                    split_config=split_cache[model_config.split_id],
                )
                signal.alarm(0)
                trained_model_count += 1
                logging.info(f"Successfully trained {trained_model_count} models")
            except TimeoutError as e:
                logging.warning(
                    f"Model fitting failed due to timeout. "
                    f"Current timeout setting: {MAX_FIT_TIME / 60:.2f} minutes. "
                    f"model_config = {model_config}"
                )

            put_result(r, model_config)
            logging.debug("Successfully saved model.")
        except Exception as e:
            uncaught_failures += 1
            logging.error(
                f"model loop failed. "
                f"{MAX_UNCAUGHT_FAILURES - uncaught_failures} remaining: {e}"
            )
            if uncaught_failures >= MAX_UNCAUGHT_FAILURES:
                raise

    logging.info("Worker is complete. Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=-1)
    args = parser.parse_args()

    main(worker_id=args.worker_id)
