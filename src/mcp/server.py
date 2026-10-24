"""Minimal MCP stdio server for the vision and action tools.

Speak JSON-RPC on stdin, one message per line. This process never commands hardware.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.mcp.tools import (
    assess_frame,
    decode_jpeg,
    dispatch_payload,
    escalate_to_human,
    read_vision,
    reanalyze_frame,
    reroute_ground_rescue,
)
from src.server.schemas import VisionState

PROTOCOL = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_vision",
        "description": "Read a VisionState as reasoning: target, hazard, confidence, and location.",
        "inputSchema": {
            "type": "object",
            "properties": {"vision": {"type": "object"}},
            "required": ["vision"],
            "additionalProperties": False,
        },
    },
    {
        "name": "assess_frame",
        "description": "Measure whether a JPEG frame is blurred.",
        "inputSchema": {
            "type": "object",
            "properties": {"image_base64": {"type": "string"}},
            "required": ["image_base64"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reanalyze_frame",
        "description": "Run the live detector on a JPEG again and return a new VisionState.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string"},
                "frame_id": {"type": "integer"},
                "timestamp": {"type": "string"},
            },
            "required": ["image_base64", "frame_id", "timestamp"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dispatch_payload",
        "description": "Record a payload dispatch. Does not fly a drone.",
        "inputSchema": {
            "type": "object",
            "properties": {"vision": {"type": "object"}},
            "required": ["vision"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reroute_ground_rescue",
        "description": "Record a ground-rescue reroute. Does not move a vehicle.",
        "inputSchema": {
            "type": "object",
            "properties": {"vision": {"type": "object"}},
            "required": ["vision"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record human review. Does not clear an existing mission.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vision": {"type": "object"},
                "blurred": {"type": "boolean"},
            },
            "required": ["vision"],
            "additionalProperties": False,
        },
    },
]


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    if message.get("id") is None:
        return None
    if method == "initialize":
        return _result(
            message,
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "aegis-vision", "version": "0.1.0"},
            },
        )
    if method == "tools/list":
        return _result(message, {"tools": TOOLS})
    if method == "tools/call":
        return _result(message, _call(message.get("params") or {}))
    return _error(message, -32601, "Method not found")


def _call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    try:
        payload = _invoke(str(name), arguments)
    except (ValueError, KeyError, TypeError) as error:
        return {"content": [{"type": "text", "text": str(error)}], "isError": True}
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "isError": False,
    }


def _invoke(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "read_vision":
        return read_vision(_vision(arguments))
    if name == "assess_frame":
        return assess_frame(decode_jpeg(arguments["image_base64"]))
    if name == "reanalyze_frame":
        state = reanalyze_frame(
            decode_jpeg(arguments["image_base64"]),
            int(arguments["frame_id"]),
            datetime.fromisoformat(arguments["timestamp"]),
        )
        return state.model_dump(mode="json")
    if name == "dispatch_payload":
        return dispatch_payload(_vision(arguments))
    if name == "reroute_ground_rescue":
        return reroute_ground_rescue(_vision(arguments))
    if name == "escalate_to_human":
        return escalate_to_human(_vision(arguments), blurred=bool(arguments.get("blurred", False)))
    raise ValueError(f"Unknown tool {name}")


def _vision(arguments: dict[str, Any]) -> VisionState:
    return VisionState.model_validate(arguments["vision"])


def _result(message: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message.get("id"), "result": result}


def _error(message: dict[str, Any], code: int, text: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": code, "message": text}}


def main() -> None:
    import sys

    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle_message(json.loads(line))
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
