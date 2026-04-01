"""Hardware-Aware Operator Cost Model — SQLite + XGBoost core package."""

from ocm.database import (
    DEFAULT_DB_PATH,
    delete_param_template,
    export_filename_suffix,
    get_connection,
    get_model_row,
    get_param_template_by_name,
    init_db,
    insert_record,
    list_models_for_op_device,
    list_op_device_pairs,
    list_param_templates,
    list_record_export_keys,
    fetch_records,
    save_param_template,
    upsert_model,
)
from ocm.features import params_to_feature_row, build_training_matrix
from ocm.keys import make_feature_order_key
from ocm.train import MIN_SAMPLES_DEFAULT, fit_and_store_model
from ocm.inference import predict_latency
from ocm.workflow import add_record_maybe_autofit

__all__ = [
    "DEFAULT_DB_PATH",
    "get_connection",
    "init_db",
    "insert_record",
    "list_op_device_pairs",
    "list_record_export_keys",
    "list_models_for_op_device",
    "fetch_records",
    "get_model_row",
    "upsert_model",
    "export_filename_suffix",
    "list_param_templates",
    "get_param_template_by_name",
    "save_param_template",
    "delete_param_template",
    "make_feature_order_key",
    "params_to_feature_row",
    "build_training_matrix",
    "MIN_SAMPLES_DEFAULT",
    "fit_and_store_model",
    "predict_latency",
    "add_record_maybe_autofit",
]
