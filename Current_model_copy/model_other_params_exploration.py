# =============================================================================
# Model for Monte Carlo exploration of non-transfer-rate parameters
# Used by run_MC_other_params.py
# Re-exports the ODE system from model_transfer_rate_exploration.py
# (identical dynamics — only the runner varies different parameters)
# This block dynamically imports model_transfer_rate_exploration.py at runtime
# using Python's importlib machinery, rather than a normal import statement.
# =============================================================================
import os
import importlib.util

# Resolve the directory containing this file so the import works regardless
# of the current working directory when the script is run.
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the base model file in the same directory.
_model_path = os.path.join(_this_dir, "model_transfer_rate_exploration.py")

# Create a module spec from the file path. "_base_model" is just an internal
# name for Python's module registry — it does not affect how we use the module.
_spec = importlib.util.spec_from_file_location("_base_model", _model_path)

# Instantiate an empty module object from the spec (not yet populated).
_mod = importlib.util.module_from_spec(_spec)

# Execute the module source, filling _mod with all classes and functions
# defined in model_transfer_rate_exploration.py.
_spec.loader.exec_module(_mod)

# Re-export the three public objects so that any script importing THIS module
# can use them directly (e.g. `from model_other_params_exploration import dual_reservoir_model`)
# without needing to know about the underlying importlib machinery.
PharmacokineticModel = _mod.PharmacokineticModel
ImmuneResponse = _mod.ImmuneResponse
dual_reservoir_model = _mod.dual_reservoir_model
