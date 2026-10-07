"""本地人机对弈：python -m gomoku.web，然后打开 http://127.0.0.1:8000。"""

import argparse
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch

from .session import GameSession


STATIC = Path(__file__).with_name("static")


class App:
    def __init__(self, models_dir, device="cpu"):
        self.models_dir = Path(models_dir).resolve()
        self.device = device
        self.sessions = {}
        self.lock = threading.Lock()


    def models(self):
        return sorted(str(p.relative_to(self.models_dir)) for p in self.models_dir.rglob("*.pt")
                      if p.is_file() and p.resolve().is_relative_to(self.models_dir))

    def request(self, path, data):
        # 序列化棋局修改和推理，双击或多个标签页不会交错落子。
        with self.lock:
            if path == "/api/new":
                name = data.get("model")
                if name not in self.models():
                    raise ValueError("模型不存在，请刷新模型列表")
                session_id = data.get("session")
                if not isinstance(session_id, str) or session_id not in self.sessions:
                    if len(self.sessions) >= 32:
                        raise ValueError("打开的棋局过多，请重启服务")
                    session_id = secrets.token_urlsafe(24)
                game = GameSession(self.models_dir / name, name, data.get("human", 1),
                                   data.get("mode", "greedy"), data.get("seed", 0), self.device,
                                   search=data.get("search"))
                self.sessions[session_id] = game
            else:
                session_id = data.get("session")
                if not isinstance(session_id, str) or session_id not in self.sessions:
                    raise ValueError("棋局不存在或服务已重启，请开始新对局")
                game = self.sessions[session_id]
                if path == "/api/move":
                    game.move(data.get("action"), data.get("version"))
                elif path == "/api/undo":
                    game.undo(data.get("version"))
                elif path == "/api/export":
                    return game.export()
                elif path != "/api/state":
                    raise ValueError("未知接口")
            return {"session": session_id, **game.snapshot()}


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8", download=False):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                filename = download if isinstance(download,str) else 'gomoku-game.json'
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/models":
                return self.send(200, {"models": app.models()})
            if path == "/api/export":
                session = parse_qs(urlsplit(self.path).query).get('session', [None])[0]
                try:
                    return self.send(200, app.request(path, {'session': session}), download=True)
                except ValueError as error:
                    return self.send(400, {"error": str(error)})
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/board.js": ("board.js", "text/javascript; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in files:
                return self.send(404, {"error": "页面不存在"})
            name, content_type = files[path]
            self.send(200, (STATIC / name).read_bytes(), content_type)

        def do_POST(self):
            # JSON + 同源检查：本地服务只接收本页面的修改请求。
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                return self.send(403, {"error": "不允许跨站修改棋局"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "请使用 JSON 请求"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("请求长度无效")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("请求必须是 JSON 对象")
                result = app.request(urlsplit(self.path).path, data)
                self.send(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.send(400, {"error": str(error)})
            except Exception as error:
                print(f"请求失败：{error}", flush=True)
                self.send(500, {"error": "模型加载或推理失败，请检查服务终端中的错误信息"})

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models-dir", default="runs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(App(args.models_dir, args.device)))
    print(f"五子棋评估界面：http://127.0.0.1:{server.server_port} （Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
