from sklearn.metrics import f1_score
import sklearn.preprocessing
import anndata as ad

## VIASH START
par = {
    'input_prediction': 'resources_test/task_label_projection/cxg_immune_cell_atlas/knn.h5ad',
    'input_solution': 'resources_test/task_label_projection/cxg_immune_cell_atlas/solution.h5ad',
    'average': 'weighted',
    'output': 'output.h5ad'
}
meta = {
    'name': 'f1'
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

print("Compute F1 score", flush=True)
metric_type = [ "macro", "micro", "weighted" ]
metric_id = [ "f1_" + x for x in metric_type]
metric_value = [ f1_score(
        input_solution.obs["label"], 
        input_prediction.obs["label_pred"], 
        average=x
    ) for x in metric_type ]

print("Create output data", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": input_solution.uns["dataset_id"],
        "normalization_id": input_solution.uns["normalization_id"],
        "method_id": input_prediction.uns["method_id"],
        "metric_ids": metric_id,
        "metric_values": metric_value
    }
)

print("Write output data", flush=True)
output.write_h5ad(par['output'], compression="gzip")

