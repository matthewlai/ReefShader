import dataclasses
import functools
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np

from JaxVidFlow import gyroflow, normalize, scale, utils, video_reader

from config_block import ConfigDict

# This is the JAX part of the processing that can all be compiled (everything except the Gyroflow processing).
@functools.partial(jax.jit, static_argnames=[
    'output_for_gyroflow', 'rotation',
    'colour_norm_enabled', 'max_gain', 'temporal_smoothing',
    'gamma_enabled', 'gamma'])
def process_step1(frame, carry, output_for_gyroflow: bool, rotation: int,
                  colour_norm_enabled: bool, max_gain: float, temporal_smoothing: float,
                  gamma_enabled: bool, gamma: float) -> tuple[jnp.ndarray, jnp.ndarray]:
  if carry is None:
    last_frame_mins = None
    last_frame_maxs = None
  else:
    last_frame_mins, last_frame_maxs = carry

  assert rotation == 0 or not output_for_gyroflow, 'Gyroflow cannot handle rotated videos yet'

  if rotation != 0:
    assert rotation % 90 == 0
    times = rotation // 90
    frame = jnp.rot90(frame, k=times)

  ref = frame

  # Theoretically we should be able to go to Rec 709 (in float), then to linear, and normalize there. But at least
  # with the DJI log, it really crashes red, and doesn't look good in practice.
  # We do the normalization in log space instead, and optionally apply gamma correction into Rec709.
  # This doesn't make much mathematical sense, but produces aesthetically pleasing results.
  # frame = lut.apply_lut(frame, 'luts/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube')
  if colour_norm_enabled:
    frame, last_frame_mins, last_frame_maxs = normalize.normalize(
        img=frame, last_frame_mins=last_frame_mins, last_frame_maxs=last_frame_maxs, max_gain=max_gain, downsample_win=4,
        temporal_smoothing=0.0)
  else:
    last_frame_mins, last_frame_maxes = None, None

  if gamma_enabled:
    frame = jnp.pow(frame, gamma)

  frame_out = frame
  if output_for_gyroflow:
    frame_out = gyroflow.to_gyroflow(frame_out)
  return frame_out, ref, (last_frame_mins, last_frame_maxs)

def process_one_frame(frame: video_reader.Frame, carry, config: ConfigDict, video_path: str) -> tuple[video_reader.Frame | None, Any]:
    if carry is None:
        carry = {}

    if config['gyroflow']['underwater']:
        preset = '{ "light_refraction_coefficient": 1.33 }'
    else:
        preset = None

    if config['gyroflow']['enabled'] and config['gyroflow']['dll_path'] and ('gyroflow' not in carry or carry['gyroflow_preset'] != preset):
        carry['gyroflow_preset'] = preset
        analysis_file = gyroflow.gyroflow_create_project_file(video_path=video_path, preset=carry['gyroflow_preset'])
        carry['gyroflow'] = gyroflow.Gyroflow(
            gyroflow_project_path=analysis_file,
            gyroflow_lib_path=config['gyroflow']['dll_path'])

    if not config['gyroflow']['enabled'] and 'gyroflow' in carry:
        del carry['gyroflow']

    using_gyroflow = 'gyroflow' in carry

    step1_carry = carry['step1_carry'] if 'step1_carry' in carry else None
    new_frame_data, next_ref, step1_carry = process_step1(
        frame.data, step1_carry, output_for_gyroflow=using_gyroflow, rotation=frame.rotation,
        colour_norm_enabled=config['colour_norm']['enabled'], max_gain=config['colour_norm']['max_gain'], temporal_smoothing=config['colour_norm']['temporal_smoothing'],
        gamma_enabled=config['gamma']['enabled'], gamma=config['gamma']['gamma'])

    last_frame = carry['last_frame'] if 'last_frame' in carry else None
    last_frame_data = carry['last_frame_data'] if 'last_frame_data' in carry else None
    last_ref = carry['last_ref'] if 'last_ref' in carry else None

    carry['last_frame'] = frame
    carry['last_frame_data'] = new_frame_data
    carry['last_ref'] = next_ref
    carry['step1_carry'] = step1_carry

    if last_frame:
        if using_gyroflow:
            last_frame_data = carry['gyroflow'].process_frame(frame=last_frame_data, frame_time=last_frame.frame_time,
                                                              rotation=last_frame.rotation, delay_one_frame=True)
            if last_frame_data is None:
                return None, carry

            last_frame_data = gyroflow.from_gyroflow(last_frame_data)

        if config['output']['side_by_side'] and last_frame_data.shape == last_ref.shape:
            last_frame_data = utils.MergeSideBySide(last_frame_data, last_ref)

        return dataclasses.replace(last_frame, data=last_frame_data), carry
    else:
        return None, carry
