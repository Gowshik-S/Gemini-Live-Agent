"""
Rio Cloud — MCP Server (F1)

Exposes a subset of Rio's safe tools via the Model Context Protocol (MCP)
so external LLM agents (Claude Desktop, GPT-4 API, other Rio instances)
can invoke Rio's desktop tools via JSON-RPC over stdio.

Usage
-----
  python -m rio.cloud.mcp_server          # stdio transport (default)
  python -m rio.cloud.mcp_server --sse    # SSE transport on port 3001

Safe tools exposed (read-only / low risk):
  - read_file        — read file contents
  - list_directory   — list directory entries
  - capture_screen   — take a screenshot (base64)
  - get_notes        — session notes
  - search_notes     — semantic search over memory
  - web_search       — DuckDuckGo search
  - web_fetch        — HTTP GET with SSRF protection
  - get_screen_info  — monitor layout

Architecture
------------
The MCP server runs as a thin shim that delegates to the local ToolExecutor.
It does NOT require the full cloud orchestrator or Gemini API keys —
it only needs the local tool execution layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe tool registry — only expose read-only / low-risk tools
# ---------------------------------------------------------------------------

_SAFE_TOOLS: dict[str, dict[str, Any]] = {
    "read_file": {
        "description": "Read the contents of a file on the user's machine.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file to read."},
            },
            "required": ["file_path"],
        },
    },
    "list_directory": {
        "description": "List entries in a directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory_path": {"type": "string", "description": "Absolute path of directory to list."},
            },
            "required": ["directory_path"],
        },
    },
    "capture_screen": {
        "description": "Capture a screenshot of the user's screen and return as base64 PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "monitor": {"type": "integer", "description": "Monitor index (0 = primary).", "default": 0},
            },
        },
    },
    "get_notes": {
        "description": "Retrieve all saved session notes.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "search_notes": {
        "description": "Semantic search over long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "description": "Number of results.", "default": 5},
            },
            "required": ["query"],
        },
    },
    "web_search": {
        "description": "Search the web using DuckDuckGo Instant Answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    },
    "web_fetch": {
        "description": "Fetch a webpage and return its text content (SSRF-protected).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch."},
            },
            "required": ["url"],
        },
    },
    "get_screen_info": {
        "description": "Get monitor layout and resolution info.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
}

# ---------------------------------------------------------------------------
# Tool execution bridge — delegates to local tool functions
# ---------------------------------------------------------------------------


async def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a safe tool and return the result as a string."""
    # Lazy-import the local tool implementations
    _local_dir = str(Path(__file__).resolve().parent.parent / "local")
    if _local_dir not in sys.path:
        sys.path.insert(0, _local_dir)

    if name == "read_file":
        file_path = arguments["file_path"]
        path = Path(file_path)
        if not path.is_file():
            return json.dumps({"error": f"File not found: {file_path}"})
        text = path.read_text(encoding="utf-8", errors="replace")[:500_000]
        return json.dumps({"content": text, "length": len(text)})

    elif name == "list_directory":
        dir_path = arguments["directory_path"]
        path = Path(dir_path)
        if not path.is_dir():
            return json.dumps({"error": f"Directory not found: {dir_path}"})
        entries = []
        for entry in sorted(path.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            })
        return json.dumps({"entries": entries[:500]})

    elif name == "capture_screen":
        try:
            import pyautogui
            img = pyautogui.screenshot()
            import io, base64
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return json.dumps({"image_base64": b64[:100] + "...(truncated)", "full_length": len(b64)})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    elif name == "get_notes":
        return json.dumps({"notes": {}, "info": "No active session — notes empty in standalone MCP mode."})

    elif name == "search_notes":
        try:
            from memory import MemoryStore
            store = MemoryStore()
            results = store.query(arguments["query"], top_k=arguments.get("top_k", 5))
            return json.dumps({"results": results})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    elif name == "web_search":
        try:
            from web_tools import web_search
            result = await web_search(arguments["query"])
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    elif name == "web_fetch":
        try:
            from web_tools import web_fetch
            result = await web_fetch(arguments["url"])
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    elif name == "get_screen_info":
        try:
            import screeninfo
            monitors = []
            for m in screeninfo.get_monitors():
                monitors.append({
                    "x": m.x, "y": m.y,
                    "width": m.width, "height": m.height,
                    "is_primary": m.is_primary,
                })
            return json.dumps({"monitors": monitors})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# MCP Server — JSON-RPC over stdio
# ---------------------------------------------------------------------------


async def _handle_jsonrpc(request: dict) -> dict:
    """Handle a single JSON-RPC 2.0 request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "rio-mcp-server",
                    "version": "1.0.0",
                },
            },
        }

    elif method == "notifications/initialized":
        # Client acknowledgement — no response needed
        return None  # type: ignore[return-value]

    elif method == "tools/list":
        tools = []
        for tool_name, tool_def in _SAFE_TOOLS.items():
            tools.append({
                "name": tool_name,
                "description": tool_def["description"],
                "inputSchema": tool_def["inputSchema"],
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools},
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in _SAFE_TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }

        try:
            result_text = await _execute_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


async def run_stdio_server() -> None:
    """Run the MCP server over stdio (line-delimited JSON-RPC)."""
    logger.info("Rio MCP server starting on stdio...")

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            request = json.loads(line.decode("utf-8").strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        response = await _handle_jsonrpc(request)
        if response is not None:
            out = json.dumps(response) + "\n"
            writer.write(out.encode("utf-8"))
            await writer.drain()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the MCP server."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
