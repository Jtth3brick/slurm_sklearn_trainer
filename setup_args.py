import logging
import os
import pickle
import sqlite3
from importlib import import_module
from itertools import product, zip_longest
from typing import Dict, List, Optional, Tuple

import pandas as pd
import redis
import yaml
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from utils import SPLIT_DIR, ModelConfig, SplitConfig

DB_PATH = "read_db.sqlite3"


def get_data(
    cohorts: List[str], schema: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Retrieve the microbiome dataset for a given split (train or validate).
    If given schema, creates, deletes, and reorders columns filling with 0.

    Args:
        cohorts (List[str]): The list of cohorts to filter to
        schema (List[str], optional): List of taxon IDs defining the expected column order.
                                    If provided, ensures output matches this schema.

    Returns:
        X (pd.DataFrame): Feature matrix with taxa readcounts.
        y (pd.Series): Binary labels (0 = Healthy, 1 = UC/CD) for each sample.
    """
    # Connect to database as read only
    engine = sqlite3.connect(DB_PATH)
    engine.execute("PRAGMA query_only = 1")

    cohort_list_str = "', '".join(cohorts)

    # SQL query to get all nonzero read counts for the selected runs
    query_seq = f""" 
        SELECT g.run, g.taxon_id AS taxon, g.rpm
        FROM genomic_sequence_rpm AS g
        INNER JOIN selected_runs AS s
          ON g.run = s.run
        WHERE s.cohort IN ('{cohort_list_str}')
    """

    # Read data into pandas DataFrame
    df = pd.read_sql(query_seq, engine)

    # Pivot table to get features (taxa) as columns and make sure columns are string type
    X = df.pivot(index="run", columns="taxon", values="rpm").fillna(0)
    X.columns = [str(col) for col in X.columns]

    # If schema is provided, ensure X matches it exactly
    if schema is not None:
        # Convert schema to strings to match column names from pivot
        schema = [str(taxon) for taxon in schema]

        # Create DataFrame with all schema columns
        missing_data = pd.DataFrame(
            0,
            index=X.index,
            columns=list(set(schema) - set(X.columns)),  # type: ignore
        )

        # Combine existing and missing columns
        X = pd.concat([X, missing_data], axis=1)

        # Keep only schema columns in correct order
        X = X[schema]

    # Get labels for the runs
    query_labels = f"""
        SELECT run, 
               CASE WHEN condition = 'Healthy' THEN 0 ELSE 1 END as label
        FROM selected_runs
        WHERE cohort IN ('{cohort_list_str}')
    """
    y = pd.read_sql(query_labels, engine).set_index("run")["label"]

    # Ensure X and y have same indices in same order
    y = y.reindex(X.index)

    # Close database connection
    engine.close()

    # Ensure all values in X are float and y are int
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    X_final: pd.DataFrame = X.astype(float)
    y_final: pd.Series = y.astype(int)

    return X_final, y_final


def get_split_config(
    split_id: str,
    train_cohorts: List[str],
    validate_cohorts: List[str],
    seed: Optional[int] = None,
    train_eval=True,
    num_cv_splits: int = 5,
) -> SplitConfig:
    """
    Prepare the input data for the models

    split_id: unique identifier
    train_cohorts: used for get_data funtion
    validate_cohorts: used for get_data funtion
        CV uses the union of train_cohorts and validate_cohorts
    seed: random state for StratifiedKFold
    train: Whether to inclue train/val data
    num_cv_splits: k in StratifiedKFold. 0 indicates no cv.
    """
    logging.info(f"Setting up split {split_id}")

    if train_eval:
        X_train, y_train = get_data(train_cohorts)
        logging.info(f"Got X_train.shape={X_train.shape} for id {split_id}")

        X_val, y_val = get_data(
            validate_cohorts, schema=[col for col in X_train.columns]
        )
        logging.info(f"Got X_val.shape={X_val.shape} for id {split_id}")
    else:
        logging.info("Skipping train/eval data setup")
        X_train, y_train = None, None
        X_val, y_val = None, None

    if num_cv_splits:
        skf = StratifiedKFold(n_splits=num_cv_splits, shuffle=True, random_state=seed)
        # Convert DataFrame indices to run indices before storing
        # Use both train and validate data for cv
        cv_cohorts = list(set(train_cohorts) | set(validate_cohorts))
        X_cv, y_cv = get_data(cv_cohorts)
        run_indices = y_cv.index
        cv_indices = []
        for train_idx, test_idx in skf.split(X_cv, y_cv):
            # Convert DataFrame indices to run ids
            train_runs = [run_indices[i] for i in train_idx]
            test_runs = [run_indices[i] for i in test_idx]
            cv_indices.append((train_runs, test_runs))
        logging.info(
            f"Added cv args of {X_cv.shape=} to split config. "
            f"Included cohorts {cv_cohorts=}."
        )
    else:
        logging.info("Skipping cv data setup")
        X_cv = None
        y_cv = None
        cv_indices = None

    return SplitConfig(
        split_id=split_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_cv=X_cv,
        y_cv=y_cv,
        cv_indices=cv_indices,
    )


def get_model_configs(steps: Dict, cache_dir: Optional[str]) -> List[Pipeline]:
    """
    Creates a list pipelines from the yaml config
    """

    def import_class(module_path: str):
        module_name, class_name = module_path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)

    # Create a param_grid sklearn style for each step permutation
    pipeline_search = []
    step_names = [list(step.keys())[0] for step in steps]
    step_configs_lists = [list(step.values())[0] for step in steps]
    for step_configs in product(*step_configs_lists):
        model_steps = []
        hyperparams = {}

        for step_name, step_config in zip(step_names, step_configs):
            func = import_class(step_config["function"])
            args = step_config.get("args", {})
            model_steps.append((step_name, func(**args)))

            # Collect hyperparameters
            for hp_name, hp_values in step_config.get("hyperparams", {}).items():
                hyperparams[f"{step_name}__{hp_name}"] = hp_values

        pipeline = Pipeline(model_steps, memory=cache_dir)
        pipeline_search.append((pipeline, hyperparams))

    # Expand out each param grid
    all_pipelines = []
    for base_pipeline, param_grid in pipeline_search:
        if param_grid:
            # Generate all parameter combinations
            for params in ParameterGrid(param_grid):
                pipeline_copy = clone(base_pipeline)
                pipeline_copy.set_params(**params)
                all_pipelines.append(pipeline_copy)
        else:
            all_pipelines.append(clone(base_pipeline))

    return all_pipelines


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("setup args active")

    logging.info("retrieving config")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Check if we're caching and setup dir
    cache_dir = None
    if config["model_caching"]["enabled"]:
        cache_dir = config["model_caching"]["dir"]
        os.makedirs(cache_dir, exist_ok=True)

    # Set up split data and pipes with caching
    # We do it on in cohort chunks to minimize worker io on split_data reads
    logging.info("Setting up splits and data")
    os.makedirs(SPLIT_DIR, exist_ok=True)
    r = redis.Redis(host="localhost", port=6379, decode_responses=False)
    for i, split_id in enumerate(list(config["splits"].keys())):
        split_config = get_split_config(
            split_id=str(split_id),
            train_cohorts=config["splits"][split_id]["train"],
            validate_cohorts=config["splits"][split_id]["validate"],
            seed=config["seed"],
            train_eval=config["train_eval"],
            num_cv_splits=config["num_cv_splits"],
        )

        # Write as pickle for workers
        savepath = f"{SPLIT_DIR}/split_{split_id}.pkl"
        logging.info(f"Writing split_config to {savepath}")
        with open(savepath, "wb") as f:
            pickle.dump(split_config, f)
            os.makedirs(SPLIT_DIR, exist_ok=True)

        # Make model_configs a list of lists for interlacing so training is fair
        logging.info(f"Setting up pipeline config for {split_id}")
        nested_model_configs = []
        for pipe_name, pipe_config in config["pipe_configs"].items():
            pipelines = get_model_configs(
                steps=pipe_config["steps"], cache_dir=cache_dir
            )
            nested_model_configs.append(
                [
                    ModelConfig(
                        split_id=str(split_id),
                        pipeline_name=pipe_name,
                        _unfit_pipe=pipe,
                    )
                    for pipe in pipelines
                ]
            )

        # Interlace the different pipelines so they are trained fairly in case of early exit
        interlaced_model_configs = [
            item
            for sublist in zip_longest(*nested_model_configs)
            for item in sublist
            if item
        ]

        logging.info(
            f"Adding {len(interlaced_model_configs)} to Redis queue for split {split_id}..."
        )

        for model_config in interlaced_model_configs:
            r.lpush("model_queue", pickle.dumps(model_config))

        logging.info(
            f"Done adding args for split {i} out of {len(list(config['splits'].keys()))}"
        )

    logging.info("Args setup complete")


if __name__ == "__main__":
    main()
