"""Adapter for numpy arrays to Qt types."""

# This file is highly inspired (read: mostly copied) from araviq6. I didn't
# want to introduce it as a dependency as it depends on qimage2ndarray,
# which has strange management for different Python Qt bindings and dependencies.
# And we only need a tiny part of araviq6 anyways.

import jax
from jax import numpy as jnp
import numpy as np

from PySide6 import QtCore, QtMultimedia

import utils

class ArrayInterfaceAroundQVideoFrame:
  def __init__(self, frame: QtMultimedia.QVideoFrame):
    self.__qvideoframe = frame
    self.__array_interface__ = dict(
      shape=(frame.height(), frame.width(), 4),
      typestr="|u1",
      data=frame.bits(0),
      strides=(frame.bytesPerLine(0), 4, 1),
      version=3,
    )

  def rgba_view(self):
    return np.asarray(self)

def array_to_qvideo_frame(
    array: jnp.ndarray, frame_to_reuse: QtMultimedia.QVideoFrame | None = None) -> QtMultimedia.QVideoFrame:
  h, w, c = array.shape
  assert c == 4
  pixel_format = QtMultimedia.QVideoFrameFormat.PixelFormat.Format_RGBX8888
  frame_format = QtMultimedia.QVideoFrameFormat(QtCore.QSize(w, h), pixel_format)
  if frame_to_reuse is not None and frame_to_reuse.width() == w and frame_to_reuse.height() == h:
    frame = frame_to_reuse
  else:
    frame = QtMultimedia.QVideoFrame(frame_format)
  frame.map(QtMultimedia.QVideoFrame.MapMode.WriteOnly)
  array_interface = ArrayInterfaceAroundQVideoFrame(frame)
  rgba_view = array_interface.rgba_view()
  assert rgba_view.shape == (h, w, 4)
  rgba_view[:] = array
  frame.unmap()
  return frame
