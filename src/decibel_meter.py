"""I2C library for PCB Artists decibel meter modules.

Programming model based on:
https://pcbartists.com/product-documentation/i2c-decibel-meter-programming-manual/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

try:
	from smbus2 import SMBus
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"smbus2 is required for I2C access. Install with: pip install smbus2"
	) from exc


class FilterMode(IntEnum):
	NONE = 0
	A_WEIGHTING = 1
	C_WEIGHTING = 2


class Register(IntEnum):
	VERSION = 0x00
	ID0 = 0x01
	ID1 = 0x02
	ID2 = 0x03
	ID3 = 0x04
	SCRATCH = 0x05
	CONTROL = 0x06
	TAVG_HIGH = 0x07
	TAVG_LOW = 0x08
	RESET = 0x09
	DECIBEL = 0x0A
	MIN = 0x0B
	MAX = 0x0C
	THR_MIN = 0x0D
	THR_MAX = 0x0E
	GAIN = 0x0F
	DBHISTORY_START = 0x14
	DBHISTORY_END = 0x77
	FREQ64_START = 0x78
	FREQ64_END = 0xB7
	FREQ16_START = 0xB8
	FREQ16_END = 0xC7


@dataclass(frozen=True)
class VersionInfo:
	raw: int

	@property
	def hardware_version(self) -> int:
		return (self.raw >> 4) & 0x0F

	@property
	def firmware_version(self) -> int:
		return self.raw & 0x0F


class DecibelMeter:
	"""Driver for PCB Artists I2C decibel meter."""

	def __init__(self, bus_num: int = 1, address: int = 0x48, bus: Optional[SMBus] = None):
		self.address = address
		self._owned_bus = bus is None
		self.bus = bus if bus is not None else SMBus(bus_num)

	def close(self) -> None:
		if self._owned_bus:
			self.bus.close()

	def __enter__(self) -> "DecibelMeter":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def _read_byte(self, reg: int) -> int:
		return self.bus.read_byte_data(self.address, reg)

	def _write_byte(self, reg: int, value: int) -> None:
		self.bus.write_byte_data(self.address, reg, value & 0xFF)

	def _read_block(self, start_reg: int, count: int) -> List[int]:
		return self.bus.read_i2c_block_data(self.address, start_reg, count)

	def read_version(self) -> VersionInfo:
		return VersionInfo(self._read_byte(Register.VERSION))

	def read_device_id(self) -> int:
		raw = self._read_block(Register.ID0, 4)
		return int.from_bytes(bytes(raw), byteorder="big", signed=False)

	def read_scratch(self) -> int:
		return self._read_byte(Register.SCRATCH)

	def write_scratch(self, value: int) -> None:
		self._write_byte(Register.SCRATCH, value)

	def verify_i2c(self, test_value: int = 0x5A) -> bool:
		self.write_scratch(test_value)
		return self.read_scratch() == (test_value & 0xFF)

	def read_control(self) -> int:
		return self._read_byte(Register.CONTROL)

	def write_control(self, value: int) -> None:
		# Bits [7:6] are reserved and should remain 0.
		self._write_byte(Register.CONTROL, value & 0x3F)

	def set_filter_mode(self, mode: FilterMode) -> None:
		ctrl = self.read_control()
		ctrl &= ~(0b11 << 1)
		ctrl |= (int(mode) & 0b11) << 1
		self.write_control(ctrl)

	def enable_interrupt(self, enable: bool = True, threshold_mode: bool = False) -> None:
		ctrl = self.read_control()
		if threshold_mode:
			ctrl |= 1 << 4
		else:
			ctrl &= ~(1 << 4)
		if enable:
			ctrl |= 1 << 3
		else:
			ctrl &= ~(1 << 3)
		self.write_control(ctrl)

	def power_down(self) -> None:
		ctrl = self.read_control()
		ctrl |= 1
		self.write_control(ctrl)

	def wake_and_reset(self) -> None:
		# Required by the module docs after waking from power-down.
		self._write_byte(Register.RESET, 1 << 3)

	def set_averaging_time_ms(self, milliseconds: int) -> None:
		if not 1 <= milliseconds <= 0xFFFF:
			raise ValueError("milliseconds must be in range 1..65535")
		high = (milliseconds >> 8) & 0xFF
		low = milliseconds & 0xFF
		self._write_byte(Register.TAVG_HIGH, high)
		# Writing low byte makes the full TAVG value take effect.
		self._write_byte(Register.TAVG_LOW, low)

	def get_averaging_time_ms(self) -> int:
		high = self._read_byte(Register.TAVG_HIGH)
		low = self._read_byte(Register.TAVG_LOW)
		return (high << 8) | low

	def read_decibel(self) -> int:
		return self._read_byte(Register.DECIBEL)

	def read_min(self) -> int:
		return self._read_byte(Register.MIN)

	def read_max(self) -> int:
		return self._read_byte(Register.MAX)

	def set_thresholds(self, min_db: int, max_db: int) -> None:
		if not (0 <= min_db <= 255 and 0 <= max_db <= 255):
			raise ValueError("thresholds must be in range 0..255")
		if min_db > max_db:
			raise ValueError("min_db must be less than or equal to max_db")
		self._write_byte(Register.THR_MIN, min_db)
		self._write_byte(Register.THR_MAX, max_db)

	def set_gain(self, gain_step: int) -> None:
		if not 0 <= gain_step <= 95:
			raise ValueError("gain_step must be in range 0..95")
		self._write_byte(Register.GAIN, gain_step)

	def clear_interrupt(self) -> None:
		self._write_byte(Register.RESET, 1 << 0)

	def clear_min_max(self) -> None:
		self._write_byte(Register.RESET, 1 << 1)

	def clear_history(self) -> None:
		self._write_byte(Register.RESET, 1 << 2)

	def soft_reset(self) -> None:
		self._write_byte(Register.RESET, 1 << 3)

	def read_history(self, count: int = 100) -> List[int]:
		if not 1 <= count <= 100:
			raise ValueError("count must be in range 1..100")
		return self._read_block(Register.DBHISTORY_START, count)

	def read_freq_64_bins(self) -> List[int]:
		return self._read_block(Register.FREQ64_START, 64)

	def read_freq_16_bins(self) -> List[int]:
		return self._read_block(Register.FREQ16_START, 16)


if __name__ == "__main__":
	with DecibelMeter() as meter:
		version = meter.read_version()
		print(f"Version raw=0x{version.raw:02X} HW={version.hardware_version} FW={version.firmware_version}")
		print(f"I2C check: {'PASS' if meter.verify_i2c() else 'FAIL'}")
		print(f"Current dB: {meter.read_decibel()} dB SPL")