import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
}
meta = {"name": "gradient_xgboost_ensemble", "resources_dir": "."}
## VIASH END

train = ad.read_h5ad(par["input_train"])
test = ad.read_h5ad(par["input_test"])
if not train.var_names.equals(test.var_names):
    raise ValueError("Training and test genes must have the same order")
if not train.obs_names.is_unique or not test.obs_names.is_unique:
    raise ValueError("Cell identifiers must be unique")
if "label" not in train.obs or train.obs["label"].isna().any():
    raise ValueError("Every training cell needs a label")
if not np.isfinite(train.var["hvg_score"]).all():
    raise ValueError("HVG scores must be finite")

recipes = json.loads((Path(meta["resources_dir"]) / "recipes.json").read_text())
labels = LabelEncoder()
y_train = labels.fit_transform(train.obs["label"].astype(str))
votes = np.zeros((test.n_obs, len(labels.classes_)), dtype=np.uint8)
threads = meta.get("cpus") or 1
for recipe in recipes:
    mask = np.zeros(train.n_vars, dtype=bool)
    mask[np.argsort(train.var["hvg_score"].to_numpy())[-recipe["n_hvg"]:]] = True
    names = train.var.get("feature_name", pd.Series(train.var_names, index=train.var_names))
    mask |= (names == "AQP3").to_numpy() | (train.var_names == "AQP3")
    if not mask.any():
        raise ValueError("No expression features available")
    x_train = train.layers["normalized"][:, mask].astype(np.float32)
    x_test = test.layers["normalized"][:, mask].astype(np.float32)
    if sparse.issparse(x_train):
        x_train, x_test = sparse.csr_matrix(x_train), sparse.csr_matrix(x_test)
    if len(labels.classes_) == 1:
        predicted = np.zeros(test.n_obs, dtype=int)
    else:
        params = {k: v for k, v in recipe.items() if k not in ("n_hvg", "seed")}
        model = XGBClassifier(**params, random_state=recipe["seed"],
                              tree_method="hist", n_jobs=threads, eval_metric="mlogloss")
        model.fit(x_train, y_train)
        predicted = model.predict(x_test).astype(int)
    votes[np.arange(test.n_obs), predicted] += 1

# LabelEncoder sorts labels, so argmax resolves tied votes lexicographically.
output = ad.AnnData(
    obs=pd.DataFrame({"label_pred": labels.inverse_transform(votes.argmax(axis=1))},
                     index=test.obs_names),
    uns={"method_id": meta["name"], "dataset_id": test.uns["dataset_id"],
         "normalization_id": test.uns["normalization_id"]},
)
output.write_h5ad(par["output"], compression="gzip")
