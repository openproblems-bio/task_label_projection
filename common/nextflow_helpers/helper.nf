Map findArgumentSchema(Map config, String argument_id) {
  def argument_groups =
    (config.argument_groups ?: []) +
    [
      arguments: config.arguments ?: []
    ]

  def schema_value = argument_groups.findResult{ gr ->
    gr.arguments.find { arg ->
      arg.name == ("--" + argument_id)
    }
  }
  return schema_value
}

Boolean checkItemAllowed(String item, List include, List exclude, String includeArgName, String excludeArgName) {

  // Throw an error if both include and exclude lists are provided
  if (include != null && exclude != null) {
      throw new Exception("Cannot define both ${includeArgName} and ${excludeArgName}")
  }

  if (include) {
    return include.contains(item)
  }
  if (exclude) {
    return !exclude.contains(item)
  }

  return true
}

// helper functions for running parameterised methods.
//
// a parameterised method is a method component that is run once per named
// parameter set ("paramset"). the paramsets are defined in the method
// component's `info.variants`, or overridden with the workflow's `--paramsets`
// file. each run is tagged with `paramset_name` and `paramset` in the state,
// which end up in the score output (null for non-parameterised methods).

// collect the default paramsets from the method configs' info.variants
Map paramsetsFromVariants(List methods) {
  methods.collectEntries { method ->
    [method.config.name, method.config.info.variants ?: [:]]
  }.findAll { it.value }
}

// expand one (id, state) tuple into one copy per (method, paramset) pair,
// plus one untagged copy for the methods without paramsets
List expandParamsets(String id, Map state, Map methodParamsets) {
  def parameterisedMethods = methodParamsets.keySet() as List
  def untagged = [
    [id, state + [paramset_method: null, paramset_name: null, paramset: null, parameterised_methods: parameterisedMethods]]
  ]
  def tagged = methodParamsets.collectMany { methodName, paramsets ->
    paramsets.collect { psName, psArgs ->
      [id, state + [paramset_method: methodName, paramset_name: psName, paramset: psArgs ?: [:], parameterised_methods: parameterisedMethods]]
    }
  }
  untagged + tagged
}

// a paramset-tagged state only runs its own method; the untagged state only
// runs the methods without paramsets
Boolean methodMatchesParamset(Map state, String compName) {
  state.paramset_method
    ? compName == state.paramset_method
    : !state.parameterised_methods.contains(compName)
}

// check a (method, paramset) pair against --methods_include/--methods_exclude.
// both the plain component name and "<name>.<paramset_name>" are matched, so a
// run can include or exclude either a whole method or a single paramset.
Boolean checkMethodAllowed(String compName, String paramsetName, List include, List exclude) {
  def names = [compName]
  if (paramsetName) {
    names += compName + "." + paramsetName
  }
  if (include != null && exclude != null) {
    throw new Exception("Cannot define both methods_include and methods_exclude")
  }
  if (include) {
    return names.any { include.contains(it) }
  }
  if (exclude) {
    return !names.any { exclude.contains(it) }
  }
  return true
}
