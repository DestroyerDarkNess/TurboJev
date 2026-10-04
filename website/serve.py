"""Serve the static site with headers required by threaded WebAssembly audio."""

from argparse import ArgumentParser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class IsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


parser = ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=4173)
parser.add_argument(
    "--root",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="directory to serve (defaults to the website folder)",
)
args = parser.parse_args()
site_dir = args.root.resolve()
server = ThreadingHTTPServer(
    (args.host, args.port), partial(IsolatedHandler, directory=site_dir)
)
print(f"Serving {site_dir} at http://{args.host}:{args.port}")
server.serve_forever()
