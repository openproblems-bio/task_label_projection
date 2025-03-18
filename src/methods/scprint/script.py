import os
import sys

import anndata as ad
import numpy as np
import scprint
import torch
from huggingface_hub import hf_hub_download
from scdataloader import Preprocessor
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scprint import scPrint

## VIASH START
par = {
    "input_train": "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
    "input_test": "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
    "output": "output.h5ad",
    "model_name": "v2-medium",
    "model": None,
    "infer_matches": "linear_sum_assignment",
}
meta = {"name": "scprint"}
## VIASH END

sys.path.append(meta["resources_dir"])
from exit_codes import exit_non_applicable

print(f"====== scPRINT version {scprint.__version__} ======", flush=True)

# Set suggested PyTorch environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("\n>>> Reading input data...", flush=True)
input_train = ad.read_h5ad(par["input_train"])
input_test = ad.read_h5ad(par["input_test"])
input_test_uns = input_test.uns.copy()

print("\n>>> Preprocessing input data...", flush=True)
# store organism ontology term id
if input_train.uns["dataset_organism"] == "homo_sapiens":
    input_train.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    input_test.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
elif input_train.uns["dataset_organism"] == "mus_musculus":
    input_train.obs["organism_ontology_term_id"] = "NCBITaxon:10090"
    input_test.obs["organism_ontology_term_id"] = "NCBITaxon:10090"
else:
    exit_non_applicable(
        f"scPRINT can only be used with human and mouse data "
        f'(dataset_organism == "{input_train.uns["dataset_organism"]}")'
    )

# move data
input_train.X = input_train.layers["counts"]
input_train.var_names = input_train.var["feature_id"]
del input_train.layers["counts"]

input_test.X = input_test.layers["counts"]
input_test.var_names = input_test.var["feature_id"]
del input_test.layers["counts"]

# applying preprocessor
preprocessor = Preprocessor(
    # Lower this threshold for test datasets
    min_valid_genes_id=min(
        0.9 * input_train.n_vars, 10000
    ),  # 90% of features up to 10,000
    # Turn off cell filtering to return results for all cells
    filter_cell_by_counts=False,
    min_nnz_genes=False,
    do_postp=False,
    # Skip ontology checks
    skip_validate=True,
)
input_train = preprocessor(input_train)
input_test = preprocessor(input_test)

# loading model
model_checkpoint_file = par["model"]
if model_checkpoint_file is None:
    print(f"\n>>> Downloading '{par['model_name']}' model...", flush=True)
    model_checkpoint_file = hf_hub_download(
        repo_id="jkobject/scPRINT", filename=f"{par['model_name']}.ckpt"
    )

if torch.cuda.is_available():
    print("CUDA is available, using GPU", flush=True)
    precision = "16"
    dtype = torch.float16
    transformer = "flash"
else:
    print("CUDA is not available, using CPU", flush=True)
    precision = "32"
    dtype = torch.float32
    transformer = "normal"

print(f"Model checkpoint file: '{model_checkpoint_file}'", flush=True)

m = torch.load(model_checkpoint_file, map_location=torch.device("cpu"))
if "label_counts" in m["hyper_parameters"]:
    model = scPrint.load_from_checkpoint(
        model_checkpoint_file,
        transformer=transformer,  # Don't use this for GPUs with flashattention
        precpt_gene_emb=None,
        classes=m["hyper_parameters"]["label_counts"],
    )
else:
    model = scPrint.load_from_checkpoint(
        model_checkpoint_file,
        transformer=transformer,  # Don't use this for GPUs with flashattention
        precpt_gene_emb=None,
    )
del m

n_cores = max(16, len(os.sched_getaffinity(0)))

print(f"Using {n_cores} worker cores")
embedder = scprint.tasks.Embedder(
    batch_size=par["batch_size"],
    how="random expr",
    max_len=par["max_len"],
    add_zero_genes=0,
    num_workers=n_cores,
    doclass=True,
    doplot=False,
    precision=precision,
    dtype=dtype,
    pred_embedding=["cell_type_ontology_term_id"],
    keep_all_cls_pred=False,
    output_expression="none",
)
embedded, _ = embedder(model, input_train, cache=False)

print("\n>>> Matching predictions to labels...", flush=True)
# The labels predicted by scPRINT might be different to those in the dataset
# This step creates a mapping from the predicted labels to the dataset labels

# Get levels and values for real labels
labels = input_train.obs["label"].astype("category")
label_values = list(labels)
label_levels = sorted(list(labels.cat.categories))

# Get levels and values for predicted labels
predicted = embedded.obs["conv_pred_cell_type_ontology_term_id"].astype("category")
predicted_values = list(predicted)
predicted_levels = sorted(list(predicted.cat.categories))

# If there are any predicted labels that exactly match a dataset label we use them directly
matches = {}
lower_label_levels = [lbl.lower() for lbl in label_levels]
print("---- EXACT MATCHES ----", flush=True)
for pred in predicted_levels:
    if pred.lower() in lower_label_levels:
        matches[pred] = label_levels[lower_label_levels.index(pred.lower())]
        print(pred, flush=True)

# Remove any predicted labels that have exact matches
predicted_levels = [pred for pred in predicted_levels if pred not in matches.keys()]

# Calculate Jaccard distance between each pair of predicted and dataset labels
jaccard = np.zeros((len(label_levels), len(predicted_levels)))
combos = [(label, pred) for label in label_levels for pred in predicted_levels]

for label, pred in combos:
    labels_bin = [1 if lbl == label else 0 for lbl in label_values]
    predicted_bin = [1 if p == pred else 0 for p in predicted_values]
    label_idx = label_levels.index(label)
    predicted_idx = predicted_levels.index(pred)
    jaccard[label_idx, predicted_idx] = distance.jaccard(labels_bin, predicted_bin)

# Use linear sum assignment to match predicted labels to dataset labels based on
# the Jaccard distances. This algorithm may not match all predicted labels so
# we remove any that have been matched and repeat until all predicted labels are
# matched to a dataset label.

print("\n---- INFERRED MATCHES ----", flush=True)
print(f"{'PREDICTED': <40}{'LABEL': <40}", flush=True)


if par["infer_matches"] == "direct":
    # other version
    for i, pred in enumerate(predicted_levels):
        matches[pred] = label_levels[np.argmin(jaccard[:, i])]
        print(f"{pred: <40}{matches[pred]: <40}", flush=True)

elif par["infer_matches"] == "linear_sum_assignment":
    # previous version
    while not all(pred in matches for pred in predicted_levels):
        # Get predicted labels that have not yet been matched
        not_matched = [pred for pred in predicted_levels if pred not in matches.keys()]
        not_matched_idx = [predicted_levels.index(pred) for pred in not_matched]
        # Get assignments for currently unmatched predicted labels
        assignments = linear_sum_assignment(jaccard[:, not_matched_idx])
        # Store any new matches
        for label, pred in zip(assignments[0], assignments[1]):
            predicted_level = not_matched[pred]
            label_level = label_levels[label]
            matches[predicted_level] = label_level
            if len(predicted_level) > 39:
                predicted_level = predicted_level[:36] + "..."
            if len(label_level) > 39:
                label_level = label_level[:36] + "..."
            print(f"{predicted_level: <40}{label_level: <40}", flush=True)
else:
    raise ValueError(f"Invalid value for infer_matches: {par['infer_matches']}")
print("\n---- UNMATCHED TRUE LABELS ----", flush=True)
for lbl in lower_label_levels:
    if lbl not in [m.lower() for m in matches.values()]:
        print(lbl, flush=True)

print("\n>>> Embedding test data...", flush=True)
embedder = scprint.tasks.Embedder(
    batch_size=par["batch_size"],
    how="random expr",
    max_len=par["max_len"],
    add_zero_genes=0,
    num_workers=n_cores,
    doclass=True,
    doplot=False,
    precision=precision,
    dtype=dtype,
    pred_embedding=["cell_type_ontology_term_id"],
    keep_all_cls_pred=False,
    output_expression="none",
)
embedded_test, _ = embedder(model, input_test, cache=False)

print("\n>>> Converting predictions to labels...", flush=True)
input_test.obs["label_pred"] = embedded_test.obs[
    "conv_pred_cell_type_ontology_term_id"
].values
input_test.obs = input_test.obs.replace(dict(label_pred=matches))

print("\n>>> Storing output...", flush=True)
output = ad.AnnData(
    obs=input_test.obs[["label_pred"]],
    uns={
        "method_id": meta["name"],
        "dataset_id": input_test_uns["dataset_id"],
        "normalization_id": input_test_uns["normalization_id"],
    },
)

print("\n>>> Writing output AnnData to file...", flush=True)
output.write_h5ad(par["output"], compression="gzip")

print("\n>>> Done!", flush=True)
