import pandas as pd
import anndata as ad
import sklearn.linear_model

## VIASH START
par = {
    'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
    'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
    'output': 'output.h5ad'
}
meta = {
    'name': 'foo',
}
## VIASH END

print("Load input data", flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])

print("Fit to train data", flush=True)
classifier = sklearn.linear_model.LogisticRegression()
classifier.fit(input_train.obsm["X_pca"], input_train.obs["label"].astype(str))

print("Predict on test data", flush=True)
label_pred = classifier.predict(input_test.obsm["X_pca"])

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
