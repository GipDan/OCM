"""Hardware-Aware Operator Cost Model — SQLite + XGBoost core package."""

from ocm.database import (
    DEFAULT_DB_PATH,
    get_connection,
    init_db,
    insert_record,
    list_op_device_pairs,
    fetch_records,
    get_model_row,
    upsert_model,
)
from ocm.features import params_to_feature_row, build_training_matrix
from ocm.train import MIN_SAMPLES_DEFAULT, fit_and_store_model
from ocm.inference import predict_latency
from ocm.workflow import add_record_maybe_autofit

__all__ = [
    "DEFAULT_DB_PATH",
    "get_connection",
    "init_db",
    "insert_record",
    "list_op_device_pairs",
    "fetch_records",
    "get_model_row",
    "upsert_model",
    "params_to_feature_row",
    "build_training_matrix",
    "MIN_SAMPLES_DEFAULT",
    "fit_and_store_model",
    "predict_latency",
    "add_record_maybe_autofit",
]
