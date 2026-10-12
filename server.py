"""Web server for the LDROBOT D500 room dashboard.

Runs the lidar reader + analytics pipeline and streams JSON to the browser
over a WebSocket (~15 Hz). Serves the dashboard over HTTP so you can open it
on this PC or from your phone on the same Wi-Fi.

Run:  python server.py [COM9]
Then open the printed URL.
"""
import asyncio, json, threading, functools, os, sys, http.server, socketserver
import ipaddress, socket, subprocess
import websockets
from ld_lidar import LidarReader
from analytics import Pipeline

HTTP_PORT = 8080
WS_PORT = 8081
WEBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

clients = set()
reader_generation = None


async def safe_send(ws, msg, timeout=0.5):
    try:
        await asyncio.wait_for(ws.send(msg), timeout=timeout)
        return True
    except Exception:
        return False


def _add_lan_ip(ips, ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return
    if addr.version != 4 or addr.is_loopback or addr.is_link_local or addr.is_unspecified:
        return
    text = str(addr)
    if text not in ips:
        ips.append(text)


def _lan_ip_score(ip):
    # Prefer common home/office LANs over VPN/virtual adapters when several
    # private routes exist on Windows.
    if ip.startswith("192.168."):
        return 0
    if ip.startswith("10."):
        return 1
    if ip.startswith("172."):
        return 2
    return 3


def get_lan_ips():
    ips = []

    if os.name == "nt":
        try:
            ps = (
                "Get-NetIPConfiguration | "
                "Where-Object { $_.IPv4Address -and $_.NetAdapter.Status -eq 'Up' } | "
                "ForEach-Object { $_.IPv4Address.IPAddress }"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True, stderr=subprocess.DEVNULL, timeout=3,
            )
            for line in out.splitlines():
                _add_lan_ip(ips, line.strip())
        except Exception:
            pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        _add_lan_ip(ips, s.getsockname()[0])
        s.close()
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            _add_lan_ip(ips, info[4][0])
    except Exception:
        pass

    return sorted(ips, key=_lan_ip_score)


def start_http():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEBDIR)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", HTTP_PORT), handler)
    httpd.daemon_threads = True
    httpd.serve_forever()


async def ws_handler(ws):
    clients.add(ws)
    try:
        async for msg in ws:
            # client -> server control messages (e.g. reset background)
            try:
                cmd = json.loads(msg)
                if cmd.get("cmd") == "reset_bg":
                    PIPE.reset()
            except Exception:
                pass
    except Exception:
        pass
    finally:
        clients.discard(ws)


async def broadcaster():
    global reader_generation
    while True:
        status = READER.get_status()
        if status["generation"] != reader_generation:
            PIPE.reset()
            reader_generation = status["generation"]

        dist, intens = READER.get_arrays()
        if status["fresh"] and status["frames"] > 0:
            out = PIPE.process(dist, intens, READER.speed)
        else:
            out = {
                "pts": [],
                "wall": [],
                "fg": [],
                "objects": [],
                "n_people": 0,
                "fg_count": 0,
                "cluster_count": 0,
                "room": None,
                "bg_ready": False,
                "rpm": 0.0,
            }
        out["frames"] = status["frames"]
        out["lidar"] = status
        msg = json.dumps(out)
        if clients:
            snapshot = list(clients)
            results = await asyncio.gather(
                *[safe_send(c, msg) for c in snapshot],
                return_exceptions=True,
            )
            for client, ok in zip(snapshot, results):
                if ok is not True:
                    clients.discard(client)
        await asyncio.sleep(1 / 15)


async def main():
    threading.Thread(target=start_http, daemon=True).start()
    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT, max_size=2**20):
        ips = get_lan_ips()
        print("=" * 54)
        print("  LDROBOT D500 room dashboard is running")
        print(f"  On this PC:   http://localhost:{HTTP_PORT}")
        if ips:
            print(f"  On phone/LAN: http://{ips[0]}:{HTTP_PORT}   (same Wi-Fi)")
            for ip in ips[1:3]:
                print(f"                http://{ip}:{HTTP_PORT}")
        else:
            print("  On phone/LAN: no LAN IP detected")
        print("=" * 54)
        await broadcaster()


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM9"
    READER = LidarReader(port)
    PIPE = Pipeline()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        READER.close()
