import av
import dataclasses
import functools
import gc
import queue
import time
import traceback

import jax
from jax import numpy as jnp
import numpy as np

from JaxVidFlow import scale, video_reader, video_writer
from PySide6 import QtCore, QtMultimedia

import np_qt_adapter
import process
import utils

CODEC_CANDIDATES = {
  'hevc_videotoolbox': 'HEVC (Apple VideoToolbox)',
  'hevc_nvenc': 'HEVC (NVIDIA NVENC)',
  'hevc_vaapi': 'HEVC (VAAPI)',
  'hevc': 'HEVC (Software)',
  'h264_videotoolbox': 'H264 (Apple VideoToolbox)',
  'h264_nvenc': 'H264 (NVIDIA NVENC)',
  'h264_vaapi': 'H264 (VAAPI)',
  'h264': 'H264 (Software)',
}

@functools.cache
def AvailableCodecs() -> list[str, str]:
  ret = []
  for codec, codec_name in CODEC_CANDIDATES.items():
    if video_writer.VideoWriter.test_codec(codec):
      ret.append((codec, codec_name))
  return ret

@dataclasses.dataclass
class VideoInfo:
  width: int
  height: int
  frame_rate: float
  duration: float
  num_frames: int
  decoder_name: str

def display_w_h(old_width: int, old_height: int, width: int, height: int, rotation: int = 0) -> tuple[int, int]:
  if rotation in (90, -90, 270, -270):
    old_width, old_height = old_height, old_width
  width_ratio = width / old_width
  height_ratio = height / old_height
  resize_ratio = min(width_ratio, height_ratio)
  new_width = int(round(old_width * resize_ratio))
  new_height = int(round(old_height * resize_ratio))
  assert (new_width == width and new_height <= height) or (new_width <= width and new_height == height)
  return new_width, new_height

@functools.partial(jax.jit, static_argnames=['rotation', 'max_val'])
def convert_to_display(img: jnp.ndarray, rotation: int, max_val: int | float) -> jnp.ndarray:
  # This function converts an image to the format Qt expects (RGBA8888).
  if img.shape[2] == 3:
    img = jnp.pad(img, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=max_val)
  if rotation != 0:
    assert rotation % 90 == 0
    times = rotation // 90
    img = jnp.rot90(img, k=times)
  if max_val == 1.0:
    assert img.dtype == jnp.float32, f'Got {img.dtype}'
    return (img * 255).astype(jnp.uint8)
  elif max_val == 255:
    assert img.dtype == jnp.uint8, f'Got {img.dtype}'
    return img
  elif max_val == 65535:
    assert img.dtype == jnp.uint16, f'Got {img.dtype}'
    return jnp.right_shift(img, 8).astype(jnp.uint8)
  else:
    raise ValueError(f'What do we do with {jnp.dtype} and max_val={max_val}?')

@functools.cache
def guess_hardware_decoders() -> list[tuple[str, str]]:
  # We create a list of everything by preference first, then filter by what's available.
  candidates = [
    # On modern Macs all accelerated decodes goes through VideoToolbox.
    ('videotoolbox', 'Apple VideoToolbox'),

    # On Windows we have both vendor-specific APIs and D3D11/12 VA. Vendor-specific APIs may
    # be faster, but let's prefer D3D11/12 VA for now because it should support everything on
    # Windows, and this way we don't have to rely on vendor-specific APIs failing gracefully
    # so we can fallback. In the future if we know some APIs do fail gracefully, we can move
    # them up above these.
    ('d3d12va', 'Direct3D 12 Video Acceleration'),
    ('d3d11va', 'Direct3D 11 Video Acceleration'),

    # On Linux there's VA-API that's supported by Intel and AMD, and cuda for NVIDIA. Hopefully
    # VA-API does fail gracefully, so we put that first, and then the vendor-specific APIs.
    ('vaapi', 'Video Acceleration API'),

    ('cuda', 'NVIDIA NVDEC'),
    ('qsv', 'Intel QuickSync'),
  ]

  available = av.codec.hwaccel.hwdevices_available()
  ret = []
  for candidate in candidates:
    if candidate[0] in available:
      ret.append(candidate)
  return ret

# Set of hwaccels that we've already seen a failure for and shouldn't try every time we load a new
# video.
failed_hwaccels = set()

class VideoProcessor(QtCore.QObject):
  # frame data, frame time
  frame_decoded = QtCore.Signal(QtMultimedia.QVideoFrame, float)
  eof = QtCore.Signal()
  video_processing_done = QtCore.Signal(bool)

  new_video_info = QtCore.Signal(VideoInfo)

  def __init__(self):
    super().__init__()
    self._path = None
    self._reader = None
    self._writer = None
    self._video_info = None
    self._last_frame = None
    self._carry = None
    self._video_processing_stop_requested = False

  def _load_video_impl(self, path):
    print(f'loading {path}')
    if self._reader:
      # If we already have a reader, we force it to be deallocated first. Otherwise
      # if we are doing hardware decoding, we can run out of hardware contexts.
      self._reader = None
      gc.collect()

    if self._carry:
      self._carry = None
      gc.collect()

    decoder_name = 'Software'
    for hwaccel, hwaccel_name in guess_hardware_decoders():
      if hwaccel in failed_hwaccels:
        continue
      try:
        self._reader = video_reader.VideoReader(filename=path, hwaccel=hwaccel)
        decoder_name = hwaccel_name
        break
      except Exception as e:
        failed_hwaccels.add(hwaccel)
        print(e)
        self._reader = None

    if self._reader is None:
      # Fallback to software decode.
      self._reader = video_reader.VideoReader(filename=path)
    print(f'done loading {path}')
    return decoder_name

  @QtCore.Slot()
  def request_load_video(self, path):
    if self._path != path:
      self._path = path

      decoder_name = self._load_video_impl(path)

      # Some formats don't record number of frames, so we estimate using duration and frame rate instead
      # (assuming constant frame rate).
      num_frames = self._reader.num_frames()
      if num_frames is None or num_frames == 0:
        num_frames = round(self._reader.duration() / self._reader.frame_rate())

      self._video_info = VideoInfo(
        width=self._reader.width(),
        height=self._reader.height(),
        frame_rate=self._reader.frame_rate(),
        duration=self._reader.duration(),
        num_frames=num_frames,
        decoder_name=decoder_name,
      )

      self._carry = None

      self.new_video_info.emit(self._video_info)

  @QtCore.Slot()
  def request_one_frame(self, width, height, try_reuse_frame, do_processing, streaming_mode, configs):
    try:
      # We may end up processing multiple frames, because gyroflow delays by one frame to avoid waiting for
      # the GPU to CPU sync.
      frame = None
      while frame is None:
        if self._last_frame is not None and try_reuse_frame:
          frame = self._last_frame
        else:
          frame = next(self._reader)
          self._last_frame = frame

        if do_processing:
          frame, self._carry = process.process_one_frame(frame, self._carry, configs, self._reader.filename())

      reader_frame, frame_time, rotation, max_val = frame.data, frame.frame_time, frame.rotation, frame.max_val
      display_frame_data = jax.device_put(convert_to_display(reader_frame, rotation=rotation, max_val=max_val), jax.devices('cpu')[0])

      # Convert to QVideoFrame here because we are still in the video processor thread. This avoids blocking
      # the GUI thread while waiting for the GPU sync.
      qt_frame = np_qt_adapter.array_to_qvideo_frame(display_frame_data, None)

      self.frame_decoded.emit(qt_frame, frame_time)

      # Tell the reader what size we want for the next frame, so they can be pre-scaled. We have to do that
      # here because the frame may be rotated and we only see that here.
      w, h = display_w_h(reader_frame.shape[1], reader_frame.shape[0], width, height, rotation)
      if rotation in (-90, 90, -270, 270):
        w, h = h, w
      self._reader.set_width(w)
      self._reader.set_height(h)
    except StopIteration:
      self.eof.emit()

  @QtCore.Slot()
  def process_video(self, output_path, configs):
    self._load_video_impl(self._path)
    if configs['scaling']['width'] != -1:
      ratio = configs['scaling']['width'] / self._reader.width()
      new_width = int(ratio * self._reader.width())
      new_height = int(ratio * self._reader.height())
      if new_width % 2 == 1:
        new_width += 1
      if new_height % 2 == 1:
        new_height += 1
      self._reader.set_width(new_width)
      self._reader.set_height(new_height)
    if self._writer:
        # If we already have a writer, we force it to be deallocated first. Otherwise
        # if we are doing hardware encoding, we can run out of hardware contexts.
        self._writer = None
        gc.collect()
    print(f'Encoding {self._path} => {output_path}')
    self._writer = video_writer.VideoWriter(
        filename=output_path,
        frame_rate=self._video_info.frame_rate,
        pixfmt='yuv420p',
        codec_name=configs['encode']['codec'],
        codec_options={},
        target_bitrate=configs['encode']['bitrate'] * 1000000)
    self._video_processing_stop_requested = False
    self._carry = None
    self.process_one_frame(configs)
    
  @QtCore.Slot()
  def process_one_frame(self, configs):
    stopping = False
    if self._video_processing_stop_requested:
      stopping = True
    else:
      try:
        # We may end up processing multiple frames, because gyroflow delays by one frame to avoid waiting for
        # the GPU to CPU sync.
        frame = None
        while frame is None:
          frame = next(self._reader)
          frame, self._carry = process.process_one_frame(frame, self._carry, configs, self._reader.filename())

        reader_frame, frame_time, rotation, max_val = frame.data, frame.frame_time, frame.rotation, frame.max_val
        display_frame_data = jax.device_put(convert_to_display(reader_frame, rotation=rotation, max_val=max_val), jax.devices('cpu')[0])

        # Convert to QVideoFrame here because we are still in the video processor thread. This avoids blocking
        # the GUI thread while waiting for the GPU sync.
        qt_frame = np_qt_adapter.array_to_qvideo_frame(display_frame_data, None)

        self.frame_decoded.emit(qt_frame, frame_time)
        self._writer.add_frame(frame=reader_frame)
        self._writer.write_audio_packets(audio_packets=self._reader.audio_packets(),
                                         in_audio_stream=self._reader.audio_stream())
        # Schedule another frame decode using singleShot() so that it's possible to interrupt processing.
        QtCore.QTimer.singleShot(1, lambda: self.process_one_frame(configs))
      except StopIteration:
        stopping = True

    if stopping:
      self._writer.close()
      self._writer = None
      gc.collect()
      self.video_processing_done.emit(self._video_processing_stop_requested)

  def stop_processing_video(self):
    self._video_processing_stop_requested = True

  @QtCore.Slot()
  def request_seek_to(self, frame_time):
    if self._reader:
      self._reader.seek(frame_time)
    self._carry = None

  @QtCore.Slot()
  def unload_video(self):
    self._path = None
    self._reader = None
    self._video_info = None
    self._carry = None
