"""
Rio Cloud — MCP Client (F2)

Connects to external MCP (Model Context Protocol) tool servers and
merges their tool definitions into the orchestrator's tool list.

Supports stdio-based MCP servers (e.g. npx @modelcontextprotocol/server-github).

Usage:
    client = McpClient()
    await client.connect_server("github", "npx @modelcontextprotocol/server-github")
    tools = client.get_tools()  # merged tool functions
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_REQUEST_TIMEOUT = 30  # seconds


class McpServerConnection:
    """A single stdio-based MCP server connection."""

    def __init__(self, name: str, command: str) -> None:
        self.name = name
        self.command = command
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tools: list[dict] = []
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start the MCP server process and perform initialization."""
        parts = self.command.split()
        self._process = await asyncio.create_subprocess_exec(
            *parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

        # Initialize
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "rio-agent", "version": "1.0.0"},
        })
        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        # List tools
        tools_result = await self._send_request("tools/list", {})
        self._tools = tools_result.get("tools", [])
        log.info(
            "mcp_client.connected",
            server=self.name,
            tools=len(self._tools),
        )

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # Extract text content from MCP response
        content = result.get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts) if texts else str(result)

    async def disconnect(self) -> None:
        """Shut down the MCP server."""
        if self._reader_task:
            self._reader_task.cancel()
        if self._process and self._process.stdin:
            self._process.stdin.close()
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except Exception:
                self._process.kill()

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for response."""
        self._request_id += 1
        rid = self._request_id
        msg = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params,
        }
        future = asyncio.get_event_loop().create_future()
        self._pending[rid] = future

        data = json.dumps(msg) + "\n"
        if self._process and self._process.stdin:
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=_REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise

    async def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        data = json.dumps(msg) + "\n"
        if self._process and self._process.stdin:
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from the server's stdout."""
        try:
            while self._process and self._process.stdout:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode().strip())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                rid = msg.get("id")
                if rid is not None and rid in self._pending:
                    future = self._pending.pop(rid)
                    if "error" in msg:
                        future.set_exception(
                            RuntimeError(msg["error"].get("message", "MCP error"))
                        )
                    else:
                        future.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("mcp_client.read_loop.error", server=self.name)


class McpClient:
    """Manages multiple MCP server connections."""

    def __init__(self) -> None:
        self._servers: dict[str, McpServerConnection] = {}

    async def connect_server(self, name: str, command: str) -> None:
        """Connect to an MCP server by name and command."""
        conn = McpServerConnection(name, command)
        await conn.connect()
        self._servers[name] = conn

    async def connect_from_config(self, config: list[dict]) -> None:
        """Connect to servers defined in config.yaml.

        Expected format: [{"name": "github", "command": "npx ..."}]
        """
        for entry in config:
            name = entry.get("name", "")
            command = entry.get("command", "")
            if name and command:
                try:
                    await self.connect_server(name, command)
                except Exception as exc:
                    log.warning(
                        "mcp_client.connect_failed",
                        server=name,
                        error=str(exc),
                    )

    def get_tool_definitions(self) -> list[dict]:
        """Return all tool definitions across all connected servers."""
        tools = []
        for server in self._servers.values():
            for tool in server._tools:
                tool["_mcp_server"] = server.name
                tools.append(tool)
        return tools

    def make_tool_functions(self) -> list:
        """Create async callable functions for each MCP tool.

        Returns functions that can be added to the orchestrator's tool list.
        """
        functions = []
        for server in self._servers.values():
            for tool_def in server._tools:
                tool_name = tool_def["name"]
                server_ref = server

                async def _mcp_call(
                    _server=server_ref,
                    _name=tool_name,
                    **kwargs,
                ) -> dict:
                    try:
                        result = await _server.call_tool(_name, kwargs)
                        return {"success": True, "result": str(result)[:5000]}
                    except Exception as exc:
                        return {"success": False, "error": str(exc)}

                _mcp_call.__name__ = f"mcp_{server.name}_{tool_name}"
                _mcp_call.__doc__ = tool_def.get(
                    "description",
                    f"MCP tool: {tool_name} from {server.name}",
                )
                functions.append(_mcp_call)
        return functions

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for server in self._servers.values():
            await server.disconnect()
        self._servers.clear()
