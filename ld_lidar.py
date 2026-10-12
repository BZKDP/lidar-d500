"""LDROBOT D500 / LD-series lidar reader (2D 360 DTOF).

Packet (47 bytes), little-endian:
  0x54 | 0x2C | speed(2) | startAngle(2, 0.01deg) | 12x[dist(2,mm) intens(1)] | endAngle(2,0.01deg) | timestamp(2) | crc(1)

Usage:
  r = LidarReader("COM9")            # opens port, starts background thread
  scan = r.get_scan()               # -> Nx3 array [angle_deg, dist_mm, intensity], full 360 snapshot
  r.close()
"""
import serial, threading, time
from serial.tools import list_ports
import numpy as np

PACKET_LEN = 47
HEADER = 0x54
VERLEN = 0x2C
POINTS_PER_PKT = 12
LIDAR_USB_IDS = {(0x10C4, 0xEA60)}  # Silicon Labs CP210x USB to UART bridge
PORT_RETRY_SEC = 1.0
STALE_AFTER_SEC = 2.0

# CRC8 table from LDROBOT SDK
_CRC_TABLE = [
0x00,0x4d,0x9a,0xd7,0x79,0x34,0xe3,0xae,0xf2,0xbf,0x68,0x25,0x8b,0xc6,0x11,0x5c,
0xa9,0xe4,0x33,0x7e,0xd0,0x9d,0x4a,0x07,0x5b,0x16,0xc1,0x8c,0x22,0x6f,0xb8,0xf5,
0x1f,0x52,0x85,0xc8,0x66,0x2b,0xfc,0xb1,0xed,0xa0,0x77,0x3a,0x94,0xd9,0x0e,0x43,
0xb6,0xfb,0x2c,0x61,0xcf,0x82,0x55,0x18,0x44,0x09,0xde,0x93,0x3d,0x70,0xa7,0xea,
0x3e,0x73,0xa4,0xe9,0x47,0x0a,0xdd,0x90,0xcc,0x81,0x56,0x1b,0xb5,0xf8,0x2f,0x62,
0x97,0xda,0x0d,0x40,0xee,0xa3,0x74,0x39,0x65,0x28,0xff,0xb2,0x1c,0x51,0x86,0xcb,
0x21,0x6c,0xbb,0xf6,0x58,0x15,0xc2,0x8f,0xd3,0x9e,0x49,0x04,0xaa,0xe7,0x30,0x7d,
0x88,0xc5,0x12,0x5f,0xf1,0xbc,0x6b,0x26,0x7a,0x37,0xe0,0xad,0x03,0x4e,0x99,0xd4,
0x7c,0x31,0xe6,0xab,0x05,0x48,0x9f,0xd2,0x8e,0xc3,0x14,0x59,0xf7,0xba,0x6d,0x20,
0xd5,0x98,0x4f,0x02,0xac,0xe1,0x36,0x7b,0x27,0x6a,0xbd,0xf0,0x5e,0x13,0xc4,0x89,
0x63,0x2e,0xf9,0xb4,0x1a,0x57,0x80,0xcd,0x91,0xdc,0x0b,0x46,0xe8,0xa5,0x72,0x3f,
0xca,0x87,0x50,0x1d,0xb3,0xfe,0x29,0x64,0x38,0x75,0xa2,0xef,0x41,0x0c,0xdb,0x96,
0x42,0x0f,0xd8,0x95,0x3b,0x76,0xa1,0xec,0xb0,0xfd,0x2a,0x67,0xc9,0x84,0x53,0x1e,
0xeb,0xa6,0x71,0x3c,0x92,0xdf,0x08,0x45,0x19,0x54,0x83,0xce,0x60,0x2d,0xfa,0xb7,
0x5d,0x10,0xc7,0x8a,0x24,0x69,0xbe,0xf3,0xaf,0xe2,0x35,0x78,0xd6,0x9b,0x4c,0x01,
0xf4,0xb9,0x6e,0x23,0x8d,0xc0,0x17,0x5a,0x06,0x4b,0x9c,0xd1,0x7f,0x32,0xe5,0xa8]

def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xff]
    return crc


def parse_packet(pkt: bytes, check_crc=True):
    """Return list of (angle_deg, dist_mm, intensity) or None if invalid."""
    if len(pkt) != PACKET_LEN or pkt[0] != HEADER or pkt[1] != VERLEN:
        return None
    if check_crc and crc8(pkt[:46]) != pkt[46]:
        return None
    start = (pkt[4] | (pkt[5] << 8)) * 0.01
    end = (pkt[42] | (pkt[43] << 8)) * 0.01
    span = end - start
    if span < 0:
        span += 360.0
    step = span / (POINTS_PER_PKT - 1)
    out = []
    for i in range(POINTS_PER_PKT):
        off = 6 + i * 3
        dist = pkt[off] | (pkt[off + 1] << 8)
        intens = pkt[off + 2]
        ang = (start + step * i) % 360.0
        out.append((ang, dist, intens))
    return out


class LidarReader:
    def __init__(self, port="COM9", baud=230400):
        self.port_hint = port
        self.baud = baud
        self.ser = None
        # 360 slots (1 deg) holding latest (dist, intensity)
        self._dist = np.zeros(360, dtype=np.float32)
        self._intens = np.zeros(360, dtype=np.float32)
        self._lock = threading.Lock()
        self._run = True
        self.frames = 0
        self.speed = 0.0  # rotation speed, deg/s (from packet)
        self.current_port = None
        self.connected = False
        self.state = "searching"
        self.last_error = ""
        self.last_packet_time = None
        self.last_open_time = None
        self.reconnects = 0
        self.generation = 0
        self.candidates = []
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _clear_scan_locked(self):
        self._dist.fill(0)
        self._intens.fill(0)
        self.speed = 0.0

    def _set_status(self, state, error=""):
        with self._lock:
            self.state = state
            self.last_error = str(error)[:180] if error else ""
            self.connected = state == "connected"

    def _candidate_ports(self):
        ports = list(list_ports.comports())
        by_device = {p.device: p for p in ports}

        def port_score(p):
            text = " ".join(str(v or "") for v in (
                p.device, p.name, p.description, p.manufacturer, p.hwid
            )).lower()
            if (p.vid, p.pid) in LIDAR_USB_IDS:
                return 0
            if "cp210" in text or "silicon labs" in text:
                return 1
            if "usb" in text and "uart" in text:
                return 2
            if "bluetooth" in text or "bthenum" in text:
                return 99
            return 20

        ordered = []
        hint = (self.port_hint or "").strip()
        if hint and hint.lower() not in ("auto", "*"):
            ordered.append(hint)

        scored = sorted(ports, key=lambda p: (port_score(p), p.device))
        for p in scored:
            if port_score(p) >= 99:
                continue
            if p.device not in ordered:
                ordered.append(p.device)

        with self._lock:
            self.candidates = ordered
        return ordered, by_device

    def _open_serial(self):
        candidates, _ = self._candidate_ports()
        if not candidates:
            with self._lock:
                self.current_port = None
                self.connected = False
                self.state = "searching"
                self.last_error = "no serial ports"
                self._clear_scan_locked()
            return False

        errors = []
        for port in candidates:
            if not self._run:
                return False
            try:
                ser = serial.Serial(port, self.baud, timeout=0.2)
                ser.reset_input_buffer()
            except Exception as e:
                errors.append(f"{port}: {e}")
                continue
            with self._lock:
                self.ser = ser
                self.current_port = port
                self.connected = False
                self.state = "waiting"
                self.last_error = ""
                self.last_open_time = time.time()
                self.last_packet_time = None
                self.reconnects += 1
                self.generation += 1
                self._clear_scan_locked()
            return True

        with self._lock:
            self.current_port = None
            self.connected = False
            self.state = "searching"
            self.last_error = "; ".join(errors)[-180:] if errors else "no usable serial ports"
            self._clear_scan_locked()
        return False

    def _close_serial(self, state="searching", error=""):
        with self._lock:
            ser = self.ser
            self.ser = None
            self.current_port = None
            self.connected = False
            self.state = state
            self.last_error = str(error)[:180] if error else ""
            self.last_packet_time = None
            self.last_open_time = None
            self._clear_scan_locked()
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _packet_age(self):
        if self.last_packet_time is None:
            return None
        return max(0.0, time.time() - self.last_packet_time)

    def _loop(self):
        buf = bytearray()
        while self._run:
            ser = self.ser
            if ser is None or not getattr(ser, "is_open", False):
                if not self._open_serial():
                    time.sleep(PORT_RETRY_SEC)
                buf.clear()
                continue

            try:
                data = ser.read(2048)
            except Exception as e:
                self._close_serial("searching", f"read error: {e}")
                buf.clear()
                time.sleep(PORT_RETRY_SEC)
                continue

            if not data:
                now = time.time()
                with self._lock:
                    last_packet = self.last_packet_time
                    last_open = self.last_open_time
                base = last_packet if last_packet is not None else last_open
                if base is not None and now - base > STALE_AFTER_SEC:
                    self._close_serial("searching", "no lidar packets")
                    buf.clear()
                    time.sleep(PORT_RETRY_SEC)
                continue
            buf.extend(data)
            # find & consume packets
            i = 0
            n = len(buf)
            parsed = 0
            while i <= n - PACKET_LEN:
                if buf[i] == HEADER and buf[i + 1] == VERLEN:
                    pkt = bytes(buf[i:i + PACKET_LEN])
                    pts = parse_packet(pkt)
                    if pts is not None:
                        parsed += 1
                        speed = (pkt[2] | (pkt[3] << 8))  # deg/s
                        now = time.time()
                        with self._lock:
                            self.frames += 1
                            self.speed = speed
                            self.connected = True
                            self.state = "connected"
                            self.last_error = ""
                            self.last_packet_time = now
                            for ang, dist, intens in pts:
                                if dist > 0:
                                    idx = int(ang) % 360
                                    self._dist[idx] = dist
                                    self._intens[idx] = intens
                        i += PACKET_LEN
                        continue
                i += 1
            del buf[:i]
            if len(buf) > 4096:
                del buf[:-PACKET_LEN]
            if parsed == 0:
                age = self._packet_age()
                if age is not None and age > STALE_AFTER_SEC:
                    self._close_serial("searching", "packet stream lost")
                    buf.clear()

    def get_scan(self):
        """Return Nx3 array [angle_deg, dist_mm, intensity] for valid points."""
        with self._lock:
            dist = self._dist.copy()
            intens = self._intens.copy()
        angles = np.arange(360, dtype=np.float32)
        mask = dist > 0
        return np.column_stack([angles[mask], dist[mask], intens[mask]])

    def get_arrays(self):
        """Return full 360-bin arrays (dist_mm, intens); 0 = no return at that angle."""
        with self._lock:
            return self._dist.copy(), self._intens.copy()

    def get_status(self):
        with self._lock:
            now = time.time()
            age = None
            if self.last_packet_time is not None:
                age = max(0.0, now - self.last_packet_time)
            fresh = self.connected and age is not None and age <= STALE_AFTER_SEC
            return {
                "state": self.state,
                "connected": bool(fresh),
                "fresh": bool(fresh),
                "port": self.current_port,
                "requested_port": self.port_hint,
                "baud": self.baud,
                "frames": int(self.frames),
                "rpm": round(self.speed / 6.0, 1),
                "last_packet_age_s": round(age, 2) if age is not None else None,
                "reconnects": int(self.reconnects),
                "generation": int(self.generation),
                "candidates": list(self.candidates),
                "last_error": self.last_error,
            }

    @property
    def rpm(self):
        return self.speed / 6.0  # deg/s -> rev/min

    def get_xy(self, max_range_mm=12000):
        """Return (x, y) meters, lidar at origin, 0deg = +X."""
        scan = self.get_scan()
        if len(scan) == 0:
            return np.array([]), np.array([])
        ang = np.deg2rad(scan[:, 0])
        d = scan[:, 1]
        m = (d > 0) & (d < max_range_mm)
        x = (d[m] * np.cos(ang[m])) / 1000.0
        y = (d[m] * np.sin(ang[m])) / 1000.0
        return x, y

    def close(self):
        self._run = False
        self._close_serial("stopped")
        try:
            self.t.join(timeout=1.0)
        except RuntimeError:
            pass


if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "COM9"
    r = LidarReader(port)
    print(f"Reading {port} for 3s...")
    time.sleep(3)
    scan = r.get_scan()
    print(f"status: {r.get_status()}")
    print(f"frames parsed: {r.frames}")
    print(f"points in 360 scan: {len(scan)}")
    if len(scan):
        d = scan[:, 1]
        print(f"distance mm: min={d.min():.0f} max={d.max():.0f} mean={d.mean():.0f}")
        print("sample (angle, dist_mm, intensity):")
        for row in scan[::40]:
            print(f"  {row[0]:6.1f}  {row[1]:7.0f}  {row[2]:4.0f}")
    r.close()
