#!/usr/bin/env python
# This technique is based on combined elimination for tuning GCC flags presented
# in:
#
#   "Fast and Effective Orchestration of Compiler Optimizations
#    for Automatic Performance Tuning", Pan and Eigenmann 2006.
#
# Combined elimination was designed to tune Boolean flags, as such the following
# technique currently leaves non-Boolean flags unchanged (these can be targeted
# by other techniques in a meta-tuning scenario).
#
# While the original technique enables all compiler flags and selectively
# disables those deemed to degrade performance, the technique presented below
# can start from an arbitary configuration and inverts rather disables each
# flag.
#
import copy
import logging
import argparse
from .technique import SequentialSearchTechnique, register
from opentuner.resultsdb.models import Result
from opentuner.search.objective import MinimizeTime, MinimizeSize

log = logging.getLogger(__name__)

class CombinedElimination(SequentialSearchTechnique):
  def __init__(self, *pargs, **kwargs):
    super(CombinedElimination, self).__init__(*pargs, **kwargs)

  def pop_param(self, name, param_list):
    try:
      param_list.remove(name)
    except ValueError:
      log.warning('%s not in param_list - nothing to remove' % name)

  def main_generator(self):
    objective = self.objective
    driver = self.driver
    manipulator = self.manipulator

    improvement = True
    # For some reason seed config results are not available until at least one
    # config has been yieled. As a work around, request the following random
    # config then start the search from the best known config so far.
    baseline_config = driver.get_configuration(self.manipulator.seed_config())
    yield baseline_config
    # Start CE search from the best config found so far
    baseline_config = driver.best_result.configuration

    params_to_consider = manipulator.parameters_dict(baseline_config.data)

    # Main search - continue until no further improvement made
    while improvement:
      improvement = False
      log.info('Starting main loop')
      # Explore phase - find which flags improve the baseline when inverted
      flags_with_improvement = []
      explore_configs = dict()
      log.info('Explore phase (%d flags to consider inverting)'
               % len(params_to_consider))
      # Construct list of configurations to explore
      for _, param in sorted(params_to_consider.items()):
        if not param.is_boolean():
          log.info('Skipping non-boolean parameter %s' % param.name)
          continue
        # flip the parameter in config.data and then convert to a Configuration
        # by calling get_configuration - this gives it the correct id and hash.
        cfg = manipulator.copy(baseline_config.data)
        param.op1_flip(cfg)
        cfg = driver.get_configuration(cfg)
        explore_configs[param] = cfg
        self.yield_nonblocking(cfg)
      # wait for the results
      yield None

      # Contstruct a list of configurations to exploit i.e. any that did better
      # than baseline_config.
      exploit_configs = []
      for param, cfg in sorted(explore_configs.items(), key=lamda x: x[0].name):
        pct_imp = 1 - objective.relative(cfg, baseline_config)
        if pct_imp > 0:
          improvement = True
          exploit_configs.append((param, cfg))
          log.info('%s: %s improvement' % (param.name, pct_imp))
      # Sort from lowest to highest improvement
      exploit_configs.sort(key=lambda x: x[1],
                           cmp=objective.compare,
                           reverse=True)

      # Exploit phase - decide which flags to invert permanently in the
      # baseline
      log.info('Exploit phase (%d configs to exploit)' % len(exploit_configs))
      # Always invert the flag with highest improvement
      if len(exploit_configs) > 0:
        param, cfg = exploit_configs.pop()
        log.info('Updating baseline %s' % param.name)
        params_to_consider.pop(param.name)
        baseline_config = cfg
      # Check whether inverting the remaining flags still improves the baseline
      while len(exploit_configs) > 0:
        param, cfg = exploit_configs.pop()
        # Flip the flag in the new baseline
        tmp_cfg = manipulator.copy(baseline_config.data)
        param.op1_flip(tmp_cfg)
        if tmp_cfg != cfg.data:
          # Baseline changed since explore phase - need new measurement
          cfg = driver.get_configuration(tmp_cfg)
          yield cfg
        pct_imp = 1 - objective.relative(cfg, baseline_config)
        if pct_imp > 0:
          log.info('Updating baseline %s' % param.name)
          params_to_consider.pop(param.name)
          baseline_config = cfg
        else:
          log.info('No improvement %s' % param.name)

register(CombinedElimination())
