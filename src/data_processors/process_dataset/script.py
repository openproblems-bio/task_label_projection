import sys
import random
import anndata as ad
import openproblems

## VIASH START
par = {
    'input': 'resources_test/common/cxg_immune_cell_atlas/dataset.h5ad',
    'method': 'batch',
    'seed': None,
    'obs_batch': 'batch',
    'obs_label': 'cell_type',
    'output_train': 'train.h5ad',
    'output_test': 'test.h5ad',
    'output_solution': 'solution.h5ad'
}
meta = {
    'resources_dir': 'src/tasks/label_projection/process_dataset',
    'config': 'src/tasks/label_projection/process_dataset/.config.vsh.yaml'
}
## VIASH END

# import helper functions
sys.path.append(meta['resources_dir'])
from subset_h5ad_by_format import subset_h5ad_by_format

# set seed if need be
if par["seed"]:
    print(f">> Setting seed to {par['seed']}")
    random.seed(par["seed"])

print(">> Load data", flush=True)
adata = ad.read_h5ad(par["input"])
print("input:", adata)

print(">> Load config", flush=True)
config = openproblems.project.read_viash_config(meta["config"])

print(">> Selecting test batch(es)", flush=True)
assert par["obs_batch"] in adata.obs.columns, f"Batch column {par['obs_batch']} not found in data"
batch_info = adata.obs[par["obs_batch"]]
batch_categories = list(batch_info.dtype.categories)
print("Batches found: ", batch_categories)

num_test_batches = par["num_test_batches"]
assert num_test_batches <= len(batch_categories), "Number of test batches is larger than the number of batches in the data"

test_batches = random.sample(batch_categories, num_test_batches)
is_test = [ x in test_batches for x in batch_info ]
print("Selected test batches: ", test_batches)

# subset the different adatas
print(">> Figuring which data needs to be copied to which output file", flush=True)
# use par arguments to look for label and batch value in different slots
field_rename_dict = {
    "obs": {
        "label": par["obs_label"],
        "batch": par["obs_batch"],
    }
}

print(">> Creating train data", flush=True)
output_train = subset_h5ad_by_format(
    adata[[not x for x in is_test]], 
    config,
    "output_train",
    field_rename_dict
)

print(">> Creating test data", flush=True)
output_test = subset_h5ad_by_format(
    adata[is_test],
    config,
    "output_test",
    field_rename_dict
)

print(">> Creating solution data", flush=True)
output_solution = subset_h5ad_by_format(
    adata[is_test],
    config,
    "output_solution",
    field_rename_dict
)

print(">> Writing data", flush=True)
output_train.write_h5ad(par["output_train"])
output_test.write_h5ad(par["output_test"])
output_solution.write_h5ad(par["output_solution"])
