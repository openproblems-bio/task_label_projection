include { checkItemAllowed } from "${meta.resources_dir}/helper.nf"
include { run_methods; run_metrics; extract_scores; create_metadata_files } from "${meta.resources_dir}/BenchmarkHelper.nf"

methods = [
  majority_vote,
  random_labels,
  true_labels,
  geneformer,
  knn,
  logistic_regression,
  mlp,
  naive_bayes,
  scanvi,
  scanvi_scarches,

  scgpt_finetuned.run(
    args: [model: file("s3://openproblems-work/cache/scGPT_human.zip")]
  ),
  scgpt_zeroshot.run(
    args: [model: file("s3://openproblems-work/cache/scGPT_human.zip")]
  ),
  scimilarity.run(
    args: [model: file("s3://openproblems-work/cache/scimilarity-model_v1.1.tar.gz")]
  ),
  scimilarity_knn.run(
    args: [model: file("s3://openproblems-work/cache/scimilarity-model_v1.1.tar.gz")]
  ),
  scprint,
  seurat_transferdata,
  singler,
  uce.run(
    args: [model: file("s3://openproblems-work/cache/uce-model-v5.zip")]
  ),
  xgboost
]

metrics = [
  accuracy,
  f1
]

workflow run_wf {
  take:
  input_ch

  main:

  /* RUN METHODS AND METRICS */
  score_ch = input_ch

    // extract the uns metadata from the dataset
    | extract_uns_metadata.run(
      fromState: [input: "input_solution"],
      toState: { id, output, state ->
        def outputYaml = readYaml(output.output)
        if (!outputYaml.uns) {
          throw new Exception("id '$id': No uns found in provided dataset")
        }
        state + [ dataset_uns: outputYaml.uns ]
      }
    )

    | run_methods(
      methods: methods,
      filter: {id, state, comp ->
        def norm = state.dataset_uns.normalization_id
        def pref = comp.config.info.preferred_normalization
        // if the preferred normalisation is none at all,
        // we can pass whichever dataset we want
        def norm_check = (norm == "log_cp10k" && pref == "counts") || norm == pref
        def method_check = checkItemAllowed(
          comp.config.name,
          state.methods_include,
          state.methods_exclude,
          "methods_include",
          "methods_exclude"
        )
        method_check && norm_check
      },
      fromState: {id, state, comp ->
        def new_args = [
          input_train: state.input_train,
          input_test: state.input_test
        ]
        if (comp.config.info.type == "control_method") {
          new_args.input_solution = state.input_solution
        }
        new_args
      },
      toState: {id, output, state, comp ->
        state + [
          method_id: comp.config.name,
          method_output: output.output
        ]
      }
    )

    | run_metrics(
      metrics: metrics,
      fromState: [
        input_solution: "input_solution",
        input_prediction: "method_output"
      ],
      toState: { id, output, state, comp ->
        state + [
          metric_id: comp.config.name,
          metric_output: output.output
        ]
      }  
    )

    | extract_scores(
      extract_uns_metadata_component: extract_uns_metadata
    )

  /* GENERATE METADATA FILES */
  metadata_ch = input_ch

    | create_metadata_files(
      datasetFile: "input_solution",
      // only keep one of the normalization methods
      // for generating the dataset metadata files
      filter: {id, state ->
        state.dataset_uns.normalization_id == "log_cp10k"
      },
      datasetUnsModifier: { uns ->
        def uns_ = uns.clone()
        uns_.remove("normalization_id")
        uns_
      },
      methods: methods,
      metrics: metrics,
      meta: meta,
      extract_uns_metadata_component: extract_uns_metadata
    )


  /* JOIN SCORES AND METADATA */
  output_ch = score_ch
    | mix(metadata_ch)
    | joinStates{ ids, states ->
      def mergedStates = states.inject([:]) { acc, m -> acc + m }
      [ids[0], mergedStates]
    }

  emit:
  output_ch
}

// Helper workflow to look for 'state.yaml' files recursively and
// use it to run the benchmark.
workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}