#!/usr/bin/env python3
"""Guarded EPG40-100 driver through the CR5 terminal RS485 bridge."""
import argparse
import re
import socket
import time
from dataclasses import asdict, dataclass


CR5_IP = "192.168.5.1"
DASHBOARD_PORT = 29999
TERMINAL_PORT = 60000
SLAVE_ID = 9
ACTION_CONFIRMATION = "MOVE_REAL_EPG40_100"


def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, "little")


@dataclass(frozen=True)
class GripperStatus:
    status: int
    position: int
    fault: int
    force: int
    speed: int
    temperature_c: int
    bus_voltage_v: int

    @property
    def enabled(self) -> bool:
        return bool(self.status & 0x01)

    def to_dict(self) -> dict:
        return {**asdict(self), "enabled": self.enabled}


def parse_status(frame: bytes) -> GripperStatus:
    if len(frame) != 13 or frame[:3] != bytes((SLAVE_ID, 4, 8)):
        raise RuntimeError(f"unexpected status response: {frame.hex(' ')}")
    if crc16(frame[:-2]) != frame[-2:]:
        raise RuntimeError("status response CRC mismatch")
    return GripperStatus(
        status=frame[4], position=frame[5], fault=frame[6], force=frame[7],
        speed=frame[8], temperature_c=frame[9], bus_voltage_v=frame[10],
    )


class EPG40100:
    def __init__(self, host: str = CR5_IP):
        self.host = host
        self.modbus_index = None
        self.terminal = None

    def _dashboard(self, command: str) -> str:
        with socket.create_connection((self.host, DASHBOARD_PORT), timeout=3) as sock:
            sock.sendall(command.encode())
            reply = sock.recv(4096).decode(errors="replace").strip()
        if not reply.startswith("0,"):
            raise RuntimeError(f"CR5 rejected {command}: {reply}")
        return reply

    def connect(self):
        self._dashboard("SetTerminal485(115200,8,N,1)")
        reply = self._dashboard(f"ModbusCreate(127.0.0.1,{TERMINAL_PORT},{SLAVE_ID},1)")
        match = re.search(r"\{(\d+)\}", reply)
        if not match:
            raise RuntimeError(f"missing Modbus index: {reply}")
        self.modbus_index = int(match.group(1))
        self.terminal = socket.create_connection((self.host, TERMINAL_PORT), timeout=3)
        self.terminal.settimeout(3)
        return self

    def close(self):
        if self.terminal:
            self.terminal.close()
            self.terminal = None
        if self.modbus_index is not None:
            self._dashboard(f"ModbusClose({self.modbus_index})")
            self.modbus_index = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()

    def _exchange(self, body: bytes) -> bytes:
        if not self.terminal:
            raise RuntimeError("gripper is not connected")
        frame = body + crc16(body)
        self.terminal.sendall(frame)
        reply = self.terminal.recv(256)
        if len(reply) < 5 or crc16(reply[:-2]) != reply[-2:]:
            raise RuntimeError("invalid Modbus response")
        return reply

    def status(self) -> GripperStatus:
        return parse_status(self._exchange(bytes((SLAVE_ID, 4, 0x07, 0xD0, 0, 4))))

    def _write_control(self, value: int):
        body = bytes((SLAVE_ID, 6, 0x03, 0xE8, value >> 8, value & 0xFF))
        if self._exchange(body)[:-2] != body:
            raise RuntimeError("control write was not echoed")

    def enable(self) -> GripperStatus:
        self._write_control(0x0001)
        return self.wait_stopped()

    def open(self) -> GripperStatus:
        self._write_control(0x050B)  # low force, low speed, open
        return self.wait_stopped()

    def close_gripper(self) -> GripperStatus:
        self._write_control(0x060B)  # low force, low speed, close
        return self.wait_stopped()

    def wait_stopped(self, timeout_s: float = 10) -> GripperStatus:
        deadline = time.monotonic() + timeout_s
        stable = 0
        previous = None
        while time.monotonic() < deadline:
            current = self.status()
            if current.fault:
                raise RuntimeError(f"EPG fault: 0x{current.fault:02x}")
            stable = stable + 1 if current.position == previous and current.speed == 0 else 0
            if stable >= 2:
                return current
            previous = current.position
            time.sleep(0.3)
        raise TimeoutError("gripper did not stop within timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "monitor", "enable", "open", "close", "cycle"))
    parser.add_argument("--confirmation")
    args = parser.parse_args()
    if args.command not in {"status", "monitor"} and args.confirmation != ACTION_CONFIRMATION:
        parser.error(f"motion command requires --confirmation {ACTION_CONFIRMATION}")

    with EPG40100() as gripper:
        if args.command == "monitor":
            print("Monitoring EPG; Ctrl+C stops communication.")
            while True:
                print(gripper.status().to_dict(), flush=True)
                time.sleep(0.5)
        action = {
            "status": gripper.status, "enable": gripper.enable, "open": gripper.open,
            "close": gripper.close_gripper,
        }.get(args.command)
        if action:
            print(action().to_dict())
        elif args.command == "cycle":
            print("open", gripper.open().to_dict())
            print("close", gripper.close_gripper().to_dict())
            print("open", gripper.open().to_dict())


if __name__ == "__main__":
    main()
