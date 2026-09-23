# Instructions

This is a guide on what to do after you have created a new task repository from the [task_template](https://github.com/openproblems-bio/task_template). More in depth information about how to create a new task can be found in the [OpenProblems documentation](https://openproblems.bio/documentation/create_task/getting-started/).

## Requirements

A list of required software to start developing a new task can be found in the [OpenProblems requirements](https://openproblems.bio/documentation/fundamentals/requirements).

## First things first

### `_viash.yaml`

Work through the `# Step N` comments in the `_viash.yaml` file: the task name, keywords, links, label, summary, description, thumbnail, test resources and authors. After performing the steps you can remove the comments.

### `common` submodule

If the `common` folder is empty, initialise the submodule:

```bash
git submodule update --init --recursive
```

Check the [common_resources README](README.md) for more information.

## Test resources

The OpenProblems team has provided test resources that can be used to develop the task. Download them into the `resources_test` folder with:

```bash
scripts/sync_resources.sh
```

Which resources are synced is defined in the `info.test_resources` section of `_viash.yaml`. When your task requires new test resources, let the OP team know so they can be added to the s3 bucket.

## Next steps

### API files ([docs](https://openproblems.bio/documentation/create_task/design-api/))

Update the API files in the `src/api` folder. The `file_*.yaml` files define the file formats passed between components, the `comp_*.yaml` files define the input and output of each type of component.

### Components ([docs](https://openproblems.bio/documentation/create_task/create-components/))

For each type of component there already is a first component created that you can modify. To create additional components, use the scripts in `scripts/create_component`, or run the underlying script directly:

```bash
common/scripts/create_component --name my_method --language python --type method
```

For each component:

1. Update the metadata and `.info` fields in the `config.vsh.yaml`.
2. Add any component specific arguments to the `config.vsh.yaml` file.
3. Add any additional resources that are required for the component.
4. Update the docker engine setup if additional packages are required.
5. If you know the required memory and or CPU you can adjust the nextflow `.directives.label` field. In addition if your component requires a GPU you can add the `gpu` label to the field.
6. Update the `script.py` or `script.R` file with the code for the component.

> [!NOTE]
> You can remove the comments in the `config.vsh.yaml` file after you have updated the file.

### Testing components ([docs](https://openproblems.bio/documentation/create_component/run-tests/))

You can test a single component by running the following command:

```bash
viash test /path/to/config.vsh.yaml
```

You can also test all components by running the following command:

```bash
scripts/project/test_all_components.sh
```

It is possible to customise the command in the above script by adding a `-q` argument to only perform the test on for example methods, e.g. `-q methods`.

### Dataset processor ([docs](https://openproblems.bio/documentation/create_task/dataset-processor/))

The dataset processor removes all information from a common dataset that the methods should not be able to see. From this filtered dataset several files are created that are used by the methods and metrics. This safeguards against data leakage, as a method cannot read data it is never handed.

The template contains an example processor in `src/data_processors/process_dataset` that you can modify. Be sure to update the `src/api/file_common_dataset.yaml` file with the fields required for the methods and metrics.

> [!IMPORTANT]
> When using your own datasets please advise the OpenProblems team on how to add these datasets to the s3 bucket.
> As the dataset processor should make use of the `common` datasets folder in the `resources` or `resources_test` directory.

### README

To create the task `README` file, run:

```bash
scripts/create_readme.sh
```

### Benchmarking ([docs](https://openproblems.bio/documentation/create_task/create-workflow/))

When you are finished with creating your components and dataset processor, update the `run_benchmark` workflow in `src/workflows` and the scripts in `scripts/run_benchmark` to benchmark the components. This workflow will be created together with the OpenProblems team.

You can test the benchmark locally on the test resources:

```bash
viash ns build --parallel --setup cachedbuild
scripts/run_benchmark/run_test_local.sh
```
