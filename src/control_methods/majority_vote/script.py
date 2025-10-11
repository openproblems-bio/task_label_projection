import anndata as ad
import pandas as pd

## VIASH START
par = {
    'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
    'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
    'output': 'output.h5ad'
}
meta = {
    'name': 'foo'
}
## VIASH END

print("Load data", flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])

print("Compute majority vote", flush=True)
label_pred = input_train.obs.label.value_counts().index[0]

print("Create output data", flush=True)
output = ad.AnnData(
    obs=pd.DataFrame(
        { 'label_pred': label_pred },
        index=input_test.obs.index
    ),
    uns={
        'method_id': meta['name'],
        "dataset_id": input_test.uns["dataset_id"],
        "normalization_id": input_test.uns["normalization_id"]
    }
)

print("Write output data", flush=True)
output.write_h5ad(par['output'], compression="gzip")
