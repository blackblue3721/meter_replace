#!/usr/bin/env python3
import argparse
import atexit
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from executor import BallTaskExecutor
from task_types import TaskRequest


class Handler(BaseHTTPRequestHandler):
    executor = BallTaskExecutor()

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            state = "enabled" if self.executor.allow_motion else "locked"
            return self.send_json(200, {"status": "ok", "hardware_motion": state})
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/tasks":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_384:
                raise ValueError("invalid request size")
            data = json.loads(self.rfile.read(length))
            request = TaskRequest(
                color=data["color"], mode=data.get("mode", "detect"),
                destination=data.get("destination", "box"),
            )
            source = data.get("source", "camera")
            if source not in {"camera", "image"}:
                raise ValueError("source must be 'camera' or 'image'")
            if source == "camera":
                result = self.executor.run_camera(request)
            else:
                image_path = Path(data["image_path"]).expanduser().resolve()
                if not image_path.is_file() or image_path.stat().st_size > 50_000_000:
                    raise ValueError("image_path must be an image file smaller than 50 MB")
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("image_path is not a readable image")
                result = self.executor.run(request, image)
            self.send_json(200, result)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"status": "error", "error": str(exc)})
        except PermissionError as exc:
            self.send_json(403, {"status": "locked", "error": str(exc)})
        except RuntimeError as exc:
            self.send_json(503, {"status": "unavailable", "error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {"status": "error", "error": str(exc)})

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    Handler.executor = BallTaskExecutor(allow_motion=args.allow_motion)
    Handler.executor.start()
    atexit.register(Handler.executor.close)
    print(f"OpenClaw Ball API: http://{args.host}:{args.port}")
    print(f"Hardware motion: {'ENABLED' if args.allow_motion else 'LOCKED'}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
