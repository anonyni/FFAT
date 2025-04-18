# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module parses an HJSON config file defining how/what toplevel to fuzz.

Description:
This module implements an HJSON configuration file parser to launch fuzzing
experiments on various toplevel designs contained in this repository. It is
designed to be invoked by the fuzz.py module in the root directory of this
repository. See the tests/ directory for example HJSON configuration files.
"""

import os
import sys

import hjson
import prettytable
from hwfutils.string_color import color_str_yellow as yellow

HWFUTILS_PATH = "infra/base-sim/hwfutils"
SHARED_TB_PATH = "infra/base-sim/tb"
LINE_SEP = "==================================================================="


class Config():
  """Loads and stores experiment configuration data."""
  def __init__(self, args):
    self.args = args
    self.config_filename = args.config_filename
    self.root_path = os.getenv("HW_FUZZING")
    
    # Load GCP configurations (unchanged)
    gcp_config = os.path.join(self.root_path, args.gcp_config_filename)
    with open(gcp_config, "r") as hjson_file:
      self.gcp_params = hjson.load(hjson_file)

    # Load experiment configurations
    with open(self.config_filename, "r") as hjson_file:
      cdict = hjson.load(hjson_file)
      self.experiment_name = cdict["experiment_name"]
      self.soc = cdict["soc"]
      self.toplevel = cdict["toplevel"]
      self.version = cdict["version"]
      self.tb_type = cdict["tb_type"]
      self.tb = cdict["tb"]
      self.fuzzer = cdict["fuzzer"]
      self.default_input = cdict["default_input"]
      self.instrument_dut = cdict["instrument_dut"]
      self.instrument_tb = cdict["instrument_tb"]
      self.instrument_vltrt = cdict["instrument_vltrt"]
      self.manual = cdict["manual"]
      self.run_on_gcp = cdict["run_on_gcp"]
      self.env_var_params = [cdict["model_params"]]
      self.env_var_params.append(cdict["hdl_gen_params"])
      self.env_var_params.append(cdict["fuzzer_params"])
      
      # --- NEW PARAMETERS FOR REAL DATASET FUZZING ---
      self.use_real_dataset = cdict.get("use_real_dataset", False)
      self.dataset_file = cdict.get("dataset_file", None)
      
    # Validate and print configurations
    self._validate_configs()
    if not self.args.silent:
      self._print_configs()

  # TODO(ttrippel): make sure didn't make mistakes writing config file!
  def _validate_configs(self):
    return

  def _print_configs(self):
    exp_config_table = prettytable.PrettyTable(header=False)
    exp_config_table.title = "Experiment Parameters"
    exp_config_table.field_names = ["Parameter", "Value"]

    # Add main experiment parameters
    exp_config_table.add_row(["Experiment Name", self.experiment_name])
    exp_config_table.add_row(["SoC", self.soc])
    exp_config_table.add_row(["Toplevel", self.toplevel])
    exp_config_table.add_row(["Version", self.version])
    exp_config_table.add_row(["Testbench Type", self.tb_type])
    exp_config_table.add_row(["Testbench", self.tb])
    exp_config_table.add_row(["Fuzzer", self.fuzzer])
    exp_config_table.add_row(["Default Input", self.default_input])
    exp_config_table.add_row(["Instrument DUT", self.instrument_dut])
    exp_config_table.add_row(["Instrument TB", self.instrument_tb])
    exp_config_table.add_row(["Instrument VLT-RT", self.instrument_vltrt])
    exp_config_table.add_row(["Manual", self.manual])
    exp_config_table.add_row(["Run on GCP", self.run_on_gcp])
    
    # NEW: Print real dataset parameters
    exp_config_table.add_row(["Use Real Dataset", self.use_real_dataset])
    exp_config_table.add_row(["Dataset File", self.dataset_file])
    
    # Add other parameters
    for params in [self.gcp_params] + self.env_var_params:
      for param, value in params.items():
        param = param.replace("_", " ").title()
        exp_config_table.add_row([param, value])

    exp_config_table.align = "l"
    print(yellow(exp_config_table.get_string()))
