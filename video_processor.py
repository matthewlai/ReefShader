import av
import dataclasses
import functools

import jax
from jax import numpy as jnp
import numpy as np

from JaxVidFlow import scale, video_reader
from PySide6 import QtCore

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

@functools.partial(jax.jit, static_argnames=['rotation', 'width', 'height'])
def convert_to_display(img: jnp.ndarray, rotation: int, width: int, height: int) -> jnp.ndarray:
  if rotation != 0:
      assert rotation % 90 == 0
      times = rotation // 90
      img = jnp.rot90(img, k=times)
  old_height, old_width = img.shape[:2]
  new_width, new_height = display_w_h(old_width, old_height, width, height)
  img = scale.scale_image(img, new_width=new_width, new_height=new_height)
  h_pad_total = width - new_width
  v_pad_total = height - new_height
  img = jnp.pad(img, pad_width=(
    (v_pad_total // 2, v_pad_total - v_pad_total // 2),
    (h_pad_total // 2, h_pad_total - h_pad_total // 2),
    (0, 0)
  ))
  assert img.shape[0] == height
  assert img.shape[1] == width
  return (img * 255).astype(jnp.uint8)

@functools.cache
def guess_hardware_decoders() -> list[tuple[str, str]]:
  # We create a list of everything by preference first, then filter by what's available.
  candidates = [
    # On modern Macs all accelerated decodes goes through VideoToolbox.
    ('videotoolbox', 'Apple VideoToolbox'),

    # On Windows we have both vendor-specific APIs and D3D11 VA. Vendor-specific APIs may
    # be faster, but let's prefer D3D11 VA for now because it should support everything on
    # Windows, and this way we don't have to rely on vendor-specific APIs failing gracefully
    # so we can fallback. In the future if we know some APIs do fail gracefully, we can move
    # them up above D3D11.
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
  frame_decoded = QtCore.Signal(jnp.ndarray, float)

  new_video_info = QtCore.Signal(VideoInfo)

  def __init__(self):
    super().__init__()
    self._path = None
    self._reader = None
    self._width = 100
    self._height = 100
    self._video_info = None

  @QtCore.Slot()
  def set_preview_width_height(self, width, height):
    self._width = width
    self._height = height

  @QtCore.Slot()
  def request_load_video(self, path):
    print(f'request load {path}')
    if self._path != path:
      self._path = path

      decoder_name = 'Software'
      self._reader = None
      for hwaccel, hwaccel_name in guess_hardware_decoders():
        if hwaccel in failed_hwaccels:
          continue
        try:
          print(f'ctor {hwaccel}')
          self._reader = video_reader.VideoReader(filename=path, hwaccel=hwaccel)
          print(f'ctor done')
          decoder_name = hwaccel_name
          break
        except:
          failed_hwaccels.add(hwaccel)
          self._reader = None

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

      print(f'emitting info')

      self.new_video_info.emit(self._video_info)

  @QtCore.Slot()
  def request_one_frame(self):
    try:
      frame = next(self._reader)
      reader_frame, frame_time, rotation = frame.data, frame.frame_time, frame.rotation
      frame = convert_to_display(reader_frame, rotation=rotation, width=self._width, height=self._height)
      # Convert to numpy here because we are still in the video processor thread. This avoids blocking
      # the GUI thread while waiting for the GPU sync.
      ret_frame = np.asarray(frame)
      self.frame_decoded.emit(ret_frame, frame_time)

      # Tell the reader what size we want for the next frame, so they can be pre-scaled. We have to do that
      # here because the frame may be rotated and we only see that here.
      w, h = display_w_h(reader_frame.shape[1], reader_frame.shape[0], self._width, self._height, rotation)
      if rotation in (-90, 90, -270, 270):
        w, h = h, w
      self._reader.set_width(w)
      self._reader.set_height(h)
    except StopIteration:
      self.frame_decoded.emit(None, None)

  @QtCore.Slot()
  def request_seek_to(self, frame_time):
    if self._reader:
      self._reader.seek(frame_time)

  @QtCore.Slot()
  def unload_video(self):
    self._path = None
    self._reader = None
    self._video_info = None