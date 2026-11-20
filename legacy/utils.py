import threading
import time

class DebugTimer:
	tl = threading.local()

	def __init__(self, block_name: str):
		self._block_name = block_name

	def __enter__(self):
		self._start = time.time()
		if 'active_count' in self.tl.__dict__:
			self._indent = self.tl.active_count + 1
			self.tl.active_count += 1
		else:
			self._indent = 0
			self.tl.active_count = 0

	def __exit__(self, type, value, traceback):
		time_diff = time.time() - self._start
		time_str = f'{(time_diff * 1000):.1f}ms' if time_diff < 1.0 else f'{time_diff:.1f}s'
		print(f'{'  ' * self._indent}{self._block_name}: {time_str}')
		self.tl.active_count -= 1