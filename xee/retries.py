# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Retry helpers for Earth Engine RPC calls."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time
import traceback
from typing import TypeVar

import numpy as np


logger = logging.getLogger(__name__)

T = TypeVar('T')


def robust_call(
    fn: Callable[[], T],
    catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    max_retries: int = 6,
    initial_delay: int = 500,
) -> T:
  """Execute a callable with exponential backoff and jitter.

  Args:
    fn: Callable to execute.
    catch: Exception type(s) to retry.
    max_retries: Maximum number of retry attempts.
    initial_delay: Initial retry delay in milliseconds.

  Returns:
    The return value of ``fn``.
  """
  assert max_retries >= 0
  for n in range(max_retries + 1):
    try:
      return fn()
    except catch:
      if n == max_retries:
        raise
      base_delay = initial_delay * 2**n
      jitter = np.random.randint(base_delay) if base_delay > 0 else 0
      next_delay = base_delay + jitter
      msg = (
          f'call failed, waiting {next_delay} ms before trying again '
          f'({max_retries - n} tries remaining). '
          f'Full traceback: {traceback.format_exc()}'
      )
      logger.debug(msg)
      time.sleep(1e-3 * next_delay)