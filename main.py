import datetime
import functools
import platform
import time

import jax
from jax import numpy as jnp
import numpy as np
import os
import signal
import sys
import random

from PySide6 import QtCore, QtMultimedia, QtMultimediaWidgets, QtWidgets, QtGui

import config_block
import np_qt_adapter
import video_processor

signal.signal(signal.SIGINT, signal.SIG_DFL)

# Allowed extensions for videos.
_ALLOWED_EXTENSIONS = [ 'mp4', 'mkv', 'mov', 'avi' ]

def _pretty_duration(seconds: float, total_seconds: float | None = None) -> str:
  to_convert = datetime.timedelta(seconds=seconds)
  # If provided, use total_seconds to determine format (whether to show hour or not), and use it to format seconds.
  if total_seconds is None:
    formatting_duration = to_convert
  else:
    formatting_duration = datetime.timedelta(seconds=total_seconds)
  show_hours = formatting_duration >= datetime.timedelta(hours=1)
  if show_hours:
    hours, remainder = divmod(to_convert, datetime.timedelta(hours=1))
    minutes, remainder = divmod(remainder, datetime.timedelta(minutes=1))
    seconds, remainder = divmod(remainder, datetime.timedelta(seconds=1))
    milliseconds, _ = divmod(remainder, datetime.timedelta(milliseconds=1))
    return f'{hours}:{minutes:>02}:{seconds:>02}.{milliseconds:>03}'
  else:
    minutes, remainder = divmod(to_convert, datetime.timedelta(minutes=1))
    seconds, remainder = divmod(remainder, datetime.timedelta(seconds=1))
    milliseconds, _ = divmod(remainder, datetime.timedelta(milliseconds=1))
    assert minutes <= 59
    return f'{minutes:>02}:{seconds:>02}.{milliseconds:>03}'

@functools.cache
def _monospace_font() -> QtGui.QFont:
  # It looks like different platforms require different style hints:
  # https://stackoverflow.com/questions/18896933/qt-qfont-selection-of-a-monospace-font-doesnt-work
  f = QtGui.QFont('Courier New')
  f.setStyleHint(QtGui.QFont.Monospace)
  if QtGui.QFontInfo(f).fixedPitch():
    return f
  f.setStyleHint(QtGui.QFont.TypeWriter)
  if QtGui.QFontInfo(f).fixedPitch():
    return f
  f.setFamily('courier')
  return f

@functools.cache
def _dll_extension() -> str:
  p = platform.system()
  if p == 'Windows':
    return 'dll'
  elif p == 'Darwin':
    return 'dylib'
  elif p == 'Linux':
    return 'so'
  else:
    raise ValueError(f'Unknown platform {p}')

class MainWidget(QtWidgets.QWidget):

  selected_video_changed = QtCore.Signal(str)
  request_one_frame = QtCore.Signal(int, int, bool, bool, bool, config_block.ConfigDict)
  unload_video = QtCore.Signal()
  seek_requested = QtCore.Signal(float)

  def __init__(self, app):
    super().__init__()

    self.setWindowTitle('ReefShader')

    self._settings = QtCore.QSettings()

    if self._settings.contains('window_pos') and self._settings.contains('window_size'):
      self.resize(self._settings.value('window_size'))
      self.move(self._settings.value('window_pos'))

    self._opened_files = []
    saved_opened_files = self._settings.value('opened_files', [], type=list)
    for path in saved_opened_files:
      if os.path.isfile(path):
        self._opened_files.append(path)

    self._common_prefix = ''
    self._video_info = None
    self._current_video_file = None

    self._video_processor_thread = QtCore.QThread()
    self._video_processor = video_processor.VideoProcessor()
    self._video_processor.moveToThread(self._video_processor_thread)
    app.aboutToQuit.connect(self._video_processor_thread.quit)

    # This keeps track of whether we have a frame request pending. If we do, there's no point queuing up seek signals or
    # more frame requests, because by the time the frame returns, we may want to be somewhere else already. This is mostly
    # for dragging the timeline or config sliders, which would otherwise generate a lot of seek signals, and a lot of
    # unnecessary work for the video processor. Instead, we just store where we want to seek to when the frame request
    # comes back, and that can be updated multiple times while a frame is pending. Same for if config has been changed 
    # (so we want to request another preview frame).
    self._frame_request_pending = False
    self._next_seek_to_time = None
    self._config_changed = False

    self._is_playing = False
    self._next_frame_display_time = 0.0

    # Left file list + file info.
    self._path_prefix_label = QtWidgets.QLabel('./')
    self._path_prefix_label.setWordWrap(True)
    self._file_list = QtWidgets.QListWidget()
    self._file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    add_files_button = QtWidgets.QPushButton('Add File(s)')
    add_folder_button = QtWidgets.QPushButton('Add Folder')
    self._remove_file_button = QtWidgets.QPushButton('Remove')
    self._remove_file_button.setEnabled(False)

    add_files_button.clicked.connect(self.open_files_dialog)
    add_folder_button.clicked.connect(self.open_dir_dialog)
    self._remove_file_button.clicked.connect(self.remove_file_clicked)
    self._file_list.itemSelectionChanged.connect(self.video_multi_selection_changed)
    self._file_list.currentTextChanged.connect(self.video_single_selection_changed)

    # Middle preview display.
    self._opened_file_label = QtWidgets.QLabel()

    self._preview_sink = QtMultimedia.QVideoSink()
    self._preview_video_widget = QtMultimediaWidgets.QVideoWidget()
    self._preview_video_widget.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)

    self._preview_play_stop_button = QtWidgets.QPushButton('⏵')
    self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._video_position_text = QtWidgets.QLabel('00:00')
    self._video_position_text.setFont(_monospace_font())
    self._preview_enable_checkbox = QtWidgets.QCheckBox('Preview')
    self._preview_enable_checkbox.setChecked(True)

    # Processing status and output folder.
    output_path_label = QtWidgets.QLabel('Output path (relative to file): ')
    self._output_path_field = QtWidgets.QLineEdit(self._settings.value('output_path', 'processed/'))
    self._process_button = QtWidgets.QPushButton('Process Selected')
    self._process_progress_text = QtWidgets.QLabel()
    self._process_progress_text.setFont(_monospace_font())

    # Left panel (input files and media info).
    file_list_controls_layout = QtWidgets.QHBoxLayout()
    file_list_controls_layout.addWidget(add_files_button)
    file_list_controls_layout.addWidget(add_folder_button)
    file_list_controls_layout.addWidget(self._remove_file_button)
    input_files_group = QtWidgets.QGroupBox('Input Files')
    input_files_group_v_layout = QtWidgets.QVBoxLayout(input_files_group)
    input_files_group_v_layout.addWidget(self._path_prefix_label)
    input_files_group_v_layout.addWidget(self._file_list)
    input_files_group_v_layout.addLayout(file_list_controls_layout)
    left_v_layout = QtWidgets.QVBoxLayout()
    left_v_layout.addWidget(input_files_group)
    media_info_group = QtWidgets.QGroupBox('Media Info')
    media_info_group_layout = QtWidgets.QVBoxLayout(media_info_group)
    self._media_info = QtWidgets.QLabel('')
    media_info_group_layout.addWidget(self._media_info)
    left_v_layout.addWidget(media_info_group)

    # Middle panel (preview).
    mid_v_layout = QtWidgets.QVBoxLayout()
    mid_v_layout.addWidget(self._opened_file_label)
    mid_v_layout.addWidget(self._preview_video_widget)

    self._preview_controls_container = QtWidgets.QWidget()
    preview_controls_layout = QtWidgets.QHBoxLayout(self._preview_controls_container)
    preview_controls_layout.addWidget(self._preview_play_stop_button, 0)
    preview_controls_layout.addWidget(self._frame_slider, 1)
    preview_controls_layout.addWidget(self._video_position_text, 0)
    preview_controls_layout.addWidget(self._preview_enable_checkbox)
    preview_controls_layout.setSpacing(5)
    preview_controls_layout.setAlignment(QtCore.Qt.AlignTop)
    mid_v_layout.addWidget(self._preview_controls_container)
    mid_v_layout.setAlignment(QtCore.Qt.AlignVCenter)

    # Middle lower panel (output path setting and processing controls).
    output_path_layout = QtWidgets.QHBoxLayout()
    output_path_layout.addWidget(output_path_label)
    output_path_layout.addWidget(self._output_path_field)
    output_path_layout.addWidget(self._process_button)
    output_path_layout.setSpacing(5)
    output_path_layout.setAlignment(QtCore.Qt.AlignTop)

    mid_v_layout.addLayout(output_path_layout, 0)

    mid_v_layout.addWidget(self._process_progress_text, 0)

    assert mid_v_layout.setStretchFactor(self._preview_video_widget, 5)

    # Option panels.
    option_blocks_v_layout = QtWidgets.QVBoxLayout()

    self._config_block_specs = [
      config_block.ConfigBlockSpec(
        block_name='scaling',
        display_name='Resolution Scaling',
        checkable=True,
        elements=[
          config_block.ConfigEnum(key='width', display_name='Width', default_index=0, options=[('1920', 1920), ('1280', 1280)]),
          config_block.ConfigBlockDescription(key='', display_name='', text='Height will be automatically set to preserve aspect ratio.')
        ]
      ),
      config_block.ConfigBlockSpec(
        block_name='gamma',
        display_name='Gamma (Contrast) Correction',
        checkable=True,
        elements=[
          config_block.ConfigFloat(key='gamma', display_name='Gamma Correction', default_value=1.1, min_value=0.5, max_value=2.0, resolution=0.01, places=2),
        ]
      ),
      config_block.ConfigBlockSpec(
        block_name='colour_norm',
        display_name='Colour Normalisation',
        checkable=True,
        elements=[
          config_block.ConfigFloat(key='max_gain', display_name='Max Gain', default_value=10, min_value=1, max_value=25, places=1),
          config_block.ConfigFloat(key='temporal_smoothing', display_name='Temporal Smoothing', default_value=0.95, min_value=0.0, max_value=1.0, resolution=0.001, places=3),
        ]
      ),
      config_block.ConfigBlockSpec(
        block_name='gyroflow',
        display_name='Gyroflow Stabilisation / Lens Correction',
        checkable=True,
        elements=[
          config_block.ConfigBool(key='underwater', display_name='Underwater Lens Correction', default_value=True),
          config_block.ConfigPath(key='dll_path', display_name='Gyroflow Frei0r Plugin Path -', path_filter=f'Library (*.{_dll_extension()})'),
        ]
      ),
      config_block.ConfigBlockSpec(
        block_name='output',
        display_name='Output Mode',
        checkable=False,
        elements=[
          config_block.ConfigBool(key='side_by_side', display_name='Side by side (processed/original)', default_value=False),
        ]
      ),
      config_block.ConfigBlockSpec(
        block_name='encode',
        display_name='Video Encode',
        checkable=False,
        elements=[
          config_block.ConfigEnum(key='codec', display_name='Codec', default_index=0, options=[
              ('H264', 'h264'),
              ('HEVC', 'hevc'),
              ('AV1', 'av1')
          ]),
          config_block.ConfigInt(key='bitrate', display_name='Bit Rate (mbps)', default_value=20, min_value=1, max_value=200),
        ]
      )
    ]

    self._config_blocks = [config_block.ConfigBlock(config_block_spec=spec, settings=self._settings) for spec in self._config_block_specs]

    for block in self._config_blocks:
      option_blocks_v_layout.addWidget(block)
      block.updated.connect(self.configs_changed)

    option_blocks_v_layout.addStretch(1)

    root_h_layout = QtWidgets.QHBoxLayout(self)
    root_h_layout.addLayout(left_v_layout)
    root_h_layout.addLayout(mid_v_layout)
    root_h_layout.addLayout(option_blocks_v_layout)

    assert root_h_layout.setStretchFactor(left_v_layout, 1)
    assert root_h_layout.setStretchFactor(mid_v_layout, 4)
    assert root_h_layout.setStretchFactor(option_blocks_v_layout, 1)

    # Connections
    # We appear to have a lot of duplicate connections here where we define a signal
    # solely to connect it to a slot in the video processor, and then we emit that
    # signal. This seems redundant as usually in Qt we can just call the slot directly,
    # but we can't actually do that safely here because the video processor runs in a
    # separate thread. The signal/slot delayed connections ensure synchronisation as
    # we pass through event loop boundaries.
    self.selected_video_changed.connect(self._video_processor.request_load_video)
    self.request_one_frame.connect(self._video_processor.request_one_frame)
    self.seek_requested.connect(self._video_processor.request_seek_to)
    self.unload_video.connect(self._video_processor.unload_video)
    self._video_processor.frame_decoded.connect(self.frame_received)
    self._video_processor.eof.connect(self.eof_received)
    self._video_processor.new_video_info.connect(self.update_video_info)
    self._frame_slider.sliderMoved.connect(self.frame_slider_moved)
    self._frame_slider.sliderPressed.connect(self.frame_slider_pressed)
    self._preview_play_stop_button.clicked.connect(self._play_stop_clicked)

    self._preview_enable_checkbox.checkStateChanged.connect(self.configs_changed)

    self._video_processor_thread.start()

    self._video_loaded = False

    self.configs_changed()
    self.video_multi_selection_changed()
    self.opened_files_updated()

  def closeEvent(self, event):
    self._settings.setValue('opened_files', self._opened_files)
    self._settings.setValue('output_path', self._output_path_field.text())
    self._settings.setValue('window_pos', self.pos())
    self._settings.setValue('window_size', self.size())

    all_configs = config_block.ConfigDict()
    for block in self._config_blocks:
      all_configs[block.name()] = block.to_config_dict()
    all_configs.save_to_settings(self._settings)

  @QtCore.Slot()
  def configs_changed(self):
    if self._frame_request_pending:
      self._config_changed = True
    elif self._video_loaded:
      self._request_new_frame(try_reuse_frame=True)

  @QtCore.Slot()
  def open_files_dialog(self):
    file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open one or more files', '', f'Videos ({" ".join(['*.' + ext for ext in _ALLOWED_EXTENSIONS])})')
    if file_names:
      self.add_files_impl(file_names)

  @QtCore.Slot()
  def open_dir_dialog(self):
    dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Add all files in directory', '')
    files_found = []
    if dir_path:
      for filename in os.listdir(dir_path):
        if filename[-3:].lower() in _ALLOWED_EXTENSIONS:
          files_found.append(os.path.join(dir_path, filename))
    if files_found:
      files_found = sorted(files_found)
      self.add_files_impl(files_found)

  @QtCore.Slot()
  def remove_file_clicked(self):
    single_file_selected = len(self._file_list.selectedItems()) == 1
    selected_row = self._file_list.row(self._file_list.selectedItems()[0])
    for item in self._file_list.selectedItems():
      full_path = os.path.join(self._common_prefix, item.text())
      assert full_path in self._opened_files
      self._opened_files.remove(full_path)
    self.opened_files_updated()
    if single_file_selected:
      if selected_row < self._file_list.count():
        self._file_list.setCurrentItem(self._file_list.item(selected_row))

  @QtCore.Slot()
  def video_single_selection_changed(self, filename):
    # This function deals with loading new video for preview, so it's called each time with the last
    # selected video. If the user selects over multiple videos with her mouse, this gets called every
    # time a video is added to the selection set.
    if filename:
      full_path = os.path.join(self._common_prefix, filename)
      self._opened_file_label.setText(full_path)
      self._set_playing(False)
      self.selected_video_changed.emit(full_path)
      self._video_loaded = True
      self._request_new_frame()
      self._current_video_file = full_path

  @QtCore.Slot()
  def video_multi_selection_changed(self):
    # This function deals with changes that aren't dealt with by video_single_selection_changed.
    selected_items = self._file_list.selectedItems()
    if not selected_items:
      self._remove_file_button.setEnabled(False)
      self._process_button.setEnabled(False)
      self.unload_video.emit()
      self._disable_preview()
      self._media_info.setText('')
      self._set_playing(False)
    else:
      self._remove_file_button.setEnabled(True)
      self._process_button.setEnabled(True)

  @QtCore.Slot()
  def update_video_info(self, video_info):
    self._video_info = video_info
    self._media_info.setText(
      f'Resolution: {video_info.width}x{video_info.height}\n'
      f'Frame rate: {video_info.frame_rate:.2f}\n'
      f'Duration: {_pretty_duration(video_info.duration)}\n'
      f'Num Frames: {video_info.num_frames}\n'
      f'Decoder: {video_info.decoder_name}')
    self._frame_slider.setValue(0)
    self._frame_slider.setMinimum(0)
    self._frame_slider.setMaximum(video_info.num_frames)
    self._preview_controls_container.setEnabled(True)

  @QtCore.Slot()
  def frame_received(self, frame: QtMultimedia.QVideoFrame, frame_time: float):
    now = time.time()
    last_frame_update_time = self._next_frame_display_time - 1.0 / self._video_info.frame_rate
    if now >= self._next_frame_display_time:
      self._update_preview(frame, frame_time)
    else:
      delay_ms = round((self._next_frame_display_time - now) * 1000)
      QtCore.QTimer.singleShot(delay_ms, lambda: self._update_preview(frame, frame_time))
    
    if self._is_playing:
      decode_time = now - last_frame_update_time
      max_fps = 1.0 / max(decode_time, 0.0001)
      too_slow = max_fps < self._video_info.frame_rate
      self._process_progress_text.setText(
        f'Decode/process time: {(decode_time * 1000):.2f}ms ({frame.width()}x{frame.height()})'
        f'{f" ({max_fps:.1f} FPS, {max_fps / self._video_info.frame_rate:.2}x)" if too_slow else ""}')

    self._frame_request_pending = False
    if self._next_seek_to_time is not None:
      self.seek_requested.emit(self._next_seek_to_time)
      self._request_new_frame()
      self._next_seek_to_time = None
    if self._config_changed:
      self._request_new_frame(try_reuse_frame=True)
      self._config_changed = False

  @QtCore.Slot()
  def eof_received(self):
    self._video_position_text.setText(_pretty_duration(self._video_info.duration, self._video_info.duration))
    self._frame_slider.setValue(self._frame_slider.maximum())
    self._set_playing(False)

    self._frame_request_pending = False
    if self._next_seek_to_time is not None:
      self.seek_requested.emit(self._next_seek_to_time)
      self._request_new_frame()
      self._next_seek_to_time = None

  @QtCore.Slot()
  def _update_preview(self, frame: QtMultimedia.QVideoFrame, frame_time: float) -> None:
    now = time.time()
    self._preview_video_widget.videoSink().setVideoFrame(frame)

    # Use duration to format frame time, so that if the video is over an hour, frame time is always shown
    # with the hour field.
    self._video_position_text.setText(_pretty_duration(frame_time, self._video_info.duration))

    if self._is_playing:
      frame_duration = 1.0 / self._video_info.frame_rate
      slider_value = round((frame_time / self._video_info.duration) * self._video_info.num_frames)
      self._frame_slider.setValue(slider_value)
      self._next_frame_display_time = now + frame_duration
      self._request_new_frame(streaming_mode=True)

  def _request_new_frame(self, try_reuse_frame=False, streaming_mode=False):
    self._frame_request_pending = True
    all_configs = config_block.ConfigDict()
    for block in self._config_blocks:
      all_configs[block.name()] = block.to_config_dict()
    process = self._preview_enable_checkbox.isChecked()
    self.request_one_frame.emit(self._preview_video_widget.width(), self._preview_video_widget.height(),
                                try_reuse_frame, process, streaming_mode, all_configs)

  def _schedule_seek(self, frame_time, start_playing=False):
    if self._frame_request_pending:
      self._next_seek_to_time = frame_time
    else:
      self._next_seek_to_time = None
      self.seek_requested.emit(frame_time)
      self._request_new_frame()
    if start_playing:
      self._set_playing(True)

  @QtCore.Slot()
  def frame_slider_moved(self, new_value):
    frame_time = new_value * 1.0 / self._video_info.frame_rate
    self._schedule_seek(frame_time)

  @QtCore.Slot()
  def frame_slider_pressed(self):
    frame_time = self._frame_slider.value() * 1.0 / self._video_info.frame_rate
    self._schedule_seek(frame_time)

  @QtCore.Slot()
  def _play_stop_clicked(self):
    self._set_playing(not self._is_playing)

  def _set_playing(self, playing):
    self._is_playing = playing
    self._preview_play_stop_button.setText('⏹' if playing else '⏵')
    if playing:
      if self._frame_slider.value() == self._frame_slider.maximum():
        self._frame_slider.setValue(0)
        self._schedule_seek(0.0, start_playing=True)  # This triggers a request new frame when seek happens.
      else:
        self._request_new_frame()

  def _disable_preview(self):
    blank_frame = np_qt_adapter.array_to_qvideo_frame(jnp.zeros(shape=(1, 1, 4), dtype=np.uint8))
    self._preview_video_widget.videoSink().setVideoFrame(blank_frame)
    self._frame_slider.setValue(0)
    self._preview_controls_container.setEnabled(False)
    self._video_loaded = False

  def add_files_impl(self, file_names):
    for file_name in file_names:
      if file_name not in self._opened_files:
        self._opened_files.append(os.path.normpath(file_name))
    self.opened_files_updated()

  def opened_files_updated(self):
    self._file_list.clear()
    if len(self._opened_files) == 0:
      self._remove_file_button.setEnabled(False)
    else:
      if len(self._opened_files) == 1:
        # Special case - use the parent directory as the common path.
        self._common_prefix = os.path.dirname(self._opened_files[0])
      else:
        self._common_prefix = os.path.commonpath(self._opened_files)
      self._path_prefix_label.setText(self._common_prefix)
      for file_name in self._opened_files:
        short_name = file_name[(len(self._common_prefix) + 1):]
        assert os.path.normpath(os.path.join(self._common_prefix, short_name)) == file_name
        QtWidgets.QListWidgetItem(short_name, self._file_list)    

if __name__ == "__main__":
  app = QtWidgets.QApplication([])

  app.setOrganizationName('ReefShader')
  app.setApplicationName('ReefShader')

  widget = MainWidget(app)
  widget.show()

  sys.exit(app.exec())