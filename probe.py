"""Probe LDROBOT D500 lidar: try several baud rates, detect valid LD packets.
LD protocol packet: 0x54 0x2C <speed2> <startAngle2> <12*(dist2,intens1)> <endAngle2> <ts2> <crc1> = 47 bytes.
"""
import serial, time, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM9"
BAUDS = [230400, 460800, 921600, 115200, 128000, 256000]

def score(port, baud, seconds=1.2):
    try:
        s = serial.Serial(port, baud, timeout=0.2)
    except Exception as e:
        return None, f"open error: {e}"
    time.sleep(0.05)
    s.reset_input_buffer()
    end = time.time() + seconds
    buf = bytearray()
    while time.time() < end:
        data = s.read(4096)
        if data:
            buf.extend(data)
    s.close()
    # count valid frames: 0x54 followed by 0x2C, 47 bytes apart pattern
    frames = 0
    i = 0
    while i < len(buf) - 1:
        if buf[i] == 0x54 and buf[i+1] == 0x2C:
            frames += 1
            i += 47
        else:
            i += 1
    return frames, f"{len(buf)} bytes, {frames} frames"

if __name__ == "__main__":
    print(f"Probing {PORT} ...")
    best = (0, None)
    for b in BAUDS:
        frames, msg = score(PORT, b)
        print(f"  {b:>7} baud -> {msg}")
        if frames and frames > best[0]:
            best = (frames, b)
    if best[1]:
        print(f"\nBEST: {best[1]} baud with {best[0]} frames")
    else:
        print("\nNo valid LD frames detected at any baud.")
