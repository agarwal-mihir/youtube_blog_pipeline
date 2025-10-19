"""Minimal helper to load a model and start the LM Studio server."""

from __future__ import annotations

import argparse
import os
import shlex


def run(args: list[str]) -> None:
    command = " ".join(shlex.quote(a) for a in args)
    code = os.system(command)
    if code != 0:
        raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start LM Studio server and load a model")
    parser.add_argument("model_key", help="Model identifier recognised by LM Studio")
    parser.add_argument("--port", type=int, default=1234, help="Server port")
    parser.add_argument("--cors", action="store_true", help="Enable CORS")
    parser.add_argument("--context-length", type=int, help="Optional context length")
    args = parser.parse_args(argv)

    load_cmd = ["lms", "load", args.model_key, "-y"]
    if args.context_length:
        load_cmd.extend(["--context-length", str(args.context_length)])
    run(load_cmd)

    server_cmd = ["lms", "server", "start", "--port", str(args.port)]
    if args.cors:
        server_cmd.append("--cors")
    run(server_cmd)


if __name__ == "__main__":
    main()
