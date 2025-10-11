import numpy as np
import sklearn.preprocessing
import anndata as ad

## VIASH START
par = {
    'input_prediction': 'resources_test/task_label_projection/cxg_immune_cell_atlas/knn.h5ad',
    'input_solution': 'resources_test/task_label_projection/cxg_immune_cell_atlas/solution.h5ad',
    'output': 'output.h5ad'
}
meta = {
    'name': 'accuracy'
}
## VIASH END

print("Load data", flush=True)
input_prediction = ad.read_h5ad(par['input_prediction'])
input_solution = ad.read_h5ad(par['input_solution'])

assert (input_prediction.obs_names == input_solution.obs_names).all(), "obs_names not the same in prediction and solution inputs"

print("Encode labels", flush=True)
cats = list(input_solution.obs["label"].dtype.categories) + list(input_prediction.obs["label_pred"].dtype.categories)
encoder = sklearn.preprocessing.LabelEncoder().fit(cats)
input_solution.obs["label"] = encoder.transform(input_solution.obs["label"])
input_prediction.obs["label_pred"] = encoder.transform(input_prediction.obs["label_pred"])

print("Compute prediction accuracy", flush=True)
accuracy = np.mean(input_solution.obs["label"] == input_prediction.obs["label_pred"])

print("Create output data", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": input_solution.uns["dataset_id"],
        "normalization_id": input_solution.uns["normalization_id"],
        "method_id": input_prediction.uns["method_id"],
        "metric_ids": "accuracy",
        "metric_values": accuracy
    }
)

print("Write output data", flush=True)
output.write_h5ad(par['output'], compression="gzip")

