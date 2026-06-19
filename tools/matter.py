"""
Matter light control tools for the voice assistant.
Allows control of Matter-enabled lights via python-matter-server.
"""

from typing import Dict, Any, Optional
import asyncio
import concurrent.futures
import json
import threading

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import your base Tool class - adjust path as needed
from core.tool_manager import Tool


class MatterClient:
    """Matter server client with a persistent, reused WebSocket connection.

    A single WebSocket connection is lazily established on first use and
    reused across all commands.  Responses are correlated to requests by
    ``message_id`` so that unsolicited event frames pushed by the server
    do not corrupt the request/response flow.
    """

    # Extra seconds added on top of self.timeout for the synchronous
    # future.result() call to allow the async coroutine to finish cleanly
    # (e.g. to return a "Connection timeout" error dict) before the sync
    # side gives up and returns its own timeout error.
    _SYNC_OVERHEAD: float = 5.0

    def __init__(self, host: str = "charmander.localdomain", port: int = 5580, timeout: float = 10.0):
        self.uri = f"ws://{host}:{port}/ws"
        self.timeout = timeout
        self._msg_id = 0
        self._msg_id_lock = threading.Lock()
        # Persistent WebSocket owned by the background event loop.
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        # Serialises concurrent send_command calls from different threads.
        self._cmd_lock = threading.Lock()

    def _next_msg_id(self) -> str:
        with self._msg_id_lock:
            self._msg_id += 1
            return str(self._msg_id)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the persistent background event loop, starting it if needed."""
        with self._loop_lock:
            if self._loop is None or not self._loop.is_running():
                loop = asyncio.new_event_loop()
                t = threading.Thread(
                    target=loop.run_forever,
                    daemon=True,
                    name="matter-ws-loop",
                )
                t.start()
                self._loop = loop
                self._loop_thread = t
        return self._loop

    async def _ensure_connected(self) -> None:
        """Establish (or re-establish) the WebSocket connection if needed.

        The first frame received after connecting is the server-info /
        handshake message; it is consumed here so callers never see it.
        """
        if self._ws is not None and not self._ws.closed:
            return
        self._ws = await asyncio.wait_for(
            websockets.connect(self.uri), timeout=self.timeout
        )
        # Consume the one-time server-info handshake frame.
        await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)

    async def _do_command(
        self, command: str, args: Optional[Dict], msg_id: str
    ) -> Dict[str, Any]:
        """Send a command and read frames until the one with matching
        ``message_id`` arrives.  Unsolicited event frames are discarded.

        The entire receive loop is bounded by a single ``self.timeout``
        deadline so continuous unsolicited traffic cannot cause indefinite
        spinning.
        """
        message: Dict[str, Any] = {"message_id": msg_id, "command": command}
        if args:
            message["args"] = args
        await self._ws.send(json.dumps(message))
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            frame = json.loads(raw)
            if frame.get("message_id") == msg_id:
                return frame
            # Unsolicited event / subscription message — skip and keep reading.

    async def _send_command_async(
        self, command: str, args: Optional[Dict]
    ) -> Dict[str, Any]:
        """Core async implementation with one transparent retry on connection drop."""
        msg_id = self._next_msg_id()

        # First attempt.
        try:
            await self._ensure_connected()
            return await self._do_command(command, args, msg_id)
        except asyncio.TimeoutError:
            return {"error": "Connection timeout"}
        except ConnectionRefusedError:
            return {"error": "Could not connect to Matter server"}
        except Exception:
            # Connection may have dropped — clear it and retry once.
            self._ws = None

        # Retry after reconnect.
        try:
            await self._ensure_connected()
            return await self._do_command(command, args, msg_id)
        except asyncio.TimeoutError:
            return {"error": "Connection timeout"}
        except ConnectionRefusedError:
            return {"error": "Could not connect to Matter server"}
        except Exception as e:
            return {"error": str(e)}

    def send_command(self, command: str, args: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a command synchronously.  Thread-safe; may be called from any thread."""
        if not WEBSOCKETS_AVAILABLE:
            return {"error": "websockets library not installed"}

        with self._cmd_lock:
            loop = self._get_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._send_command_async(command, args), loop
            )
            try:
                return future.result(timeout=self.timeout + self._SYNC_OVERHEAD)
            except concurrent.futures.TimeoutError:
                return {"error": "Connection timeout"}
            except Exception as e:
                return {"error": str(e)}

    def device_command(
        self,
        node_id: int,
        endpoint_id: int,
        cluster_id: int,
        command_name: str,
        payload: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Send a device command."""
        return self.send_command("device_command", {
            "node_id": node_id,
            "endpoint_id": endpoint_id,
            "cluster_id": cluster_id,
            "command_name": command_name,
            "payload": payload or {},
        })

    def get_nodes(self) -> Dict[str, Any]:
        """Get all commissioned nodes."""
        return self.send_command("get_nodes")

    def close(self) -> None:
        """Close the persistent WebSocket connection and stop the background loop."""
        with self._loop_lock:
            loop = self._loop
            ws = self._ws

        if loop is not None and loop.is_running():
            async def _close_ws() -> None:
                if ws is not None and not ws.closed:
                    await ws.close()

            try:
                asyncio.run_coroutine_threadsafe(_close_ws(), loop).result(timeout=5.0)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

        self._ws = None
        with self._loop_lock:
            self._loop = None
            self._loop_thread = None


class MatterLightControlTool(Tool):
    """Tool to control Matter-enabled lights."""

    def __init__(
        self,
        matter_host: str = "charmander.localdomain",
        matter_port: int = 5580,
        # Default device mapping - customize for your setup
        device_aliases: Optional[Dict[str, Dict[str, int]]] = None,
        groups: Optional[Dict[str, list]] = None
    ):
        """
        Initialize the Matter light control tool.

        Args:
            matter_host: Hostname of the python-matter-server
            matter_port: Port of the python-matter-server (default 5580)
            device_aliases: Optional mapping of friendly names to node/endpoint
                           e.g., {"bathroom": {"node_id": 1, "endpoint_id": 1},
                                  "bedroom_globe": {"node_id": 2, "endpoint_id": 1}}
            groups: Optional mapping of group names to lists of device names
                   e.g., {"bedroom": ["bedroom_globe", "bedroom_floor"]}
                   Groups can contain devices or other groups (nested groups supported)
        """
        self.client = MatterClient(matter_host, matter_port) if WEBSOCKETS_AVAILABLE else None
        self.device_aliases = device_aliases or {
            "light": {"node_id": 1, "endpoint_id": 1},  # Default single light
        }
        self.groups = groups or {}

    def _resolve_target(self, target: str, visited: Optional[set] = None) -> list:
        """
        Resolve a target name to a list of device info dictionaries.
        Supports both individual devices and groups (including nested groups).

        Args:
            target: Device name or group name to resolve
            visited: Set of already-visited targets to prevent circular references

        Returns:
            List of device info dicts: [{"node_id": X, "endpoint_id": Y, "name": "device_name"}, ...]
        """
        if visited is None:
            visited = set()

        # Prevent circular references
        if target in visited:
            return []
        visited.add(target)

        # Check if target is a group
        if target in self.groups:
            devices = []
            for member in self.groups[target]:
                devices.extend(self._resolve_target(member, visited.copy()))
            return devices

        # Check if target is an individual device
        if target in self.device_aliases:
            device_info = self.device_aliases[target].copy()
            device_info["name"] = target
            return [device_info]

        # Target not found
        return []

    @property
    def name(self) -> str:
        return "control_light"

    @property
    def description(self) -> str:
        devices = ", ".join(self.device_aliases.keys())
        description = f"Control Matter-enabled lights by device or group name."
        if self.groups:
            groups = ", ".join(self.groups.keys())
            description += f" Available groups: {groups}."
        description += " Can turn on/off, set brightness (0-100%), and set color temperature (warm/cool)."
        return description

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Which light or group to control. Devices: " + ", ".join([f"'{d}'" for d in self.device_aliases.keys()]) + (". Groups: " + ", ".join([f"'{g}'" for g in self.groups.keys()]) if self.groups else ""),
                    "examples": list(self.device_aliases.keys()) + list(self.groups.keys())
                },
                "action": {
                    "type": "string",
                    "enum": ["on", "off", "toggle", "set_brightness", "set_color_temp"],
                    "description": "Action to perform on the light"
                },
                "brightness": {
                    "type": "number",
                    "description": "Brightness level 0-100 (percentage). Only used with 'set_brightness' action.",
                    "minimum": 0,
                    "maximum": 100
                },
                "color_temp": {
                    "type": "string",
                    "enum": ["warm", "neutral", "cool", "daylight"],
                    "description": "Color temperature preset. Only used with 'set_color_temp' action."
                }
            },
            "required": ["action"],
            "additionalProperties": False
        }

    def execute(
        self,
        action: str,
        target: Optional[str] = None,
        brightness: Optional[float] = None,
        color_temp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a light control command on one or more devices."""

        if not WEBSOCKETS_AVAILABLE:
            return {"success": False, "error": "websockets library not installed. Run: pip install websockets"}

        if not self.client:
            return {"success": False, "error": "Matter client not initialized"}

        # Resolve target to list of devices
        target = target or "bedroom"  # Default target
        devices = self._resolve_target(target)

        if not devices:
            available = list(self.device_aliases.keys()) + list(self.groups.keys())
            return {
                "success": False,
                "error": f"Unknown device or group '{target}'. Available: {', '.join(available)}"
            }

        # Execute command on all resolved devices
        results = []
        for device_info in devices:
            result = self._execute_single_device(
                device_info=device_info,
                action=action,
                brightness=brightness,
                color_temp=color_temp
            )
            results.append(result)

        # Aggregate results
        return self._aggregate_results(results, target, action, brightness, color_temp)

    def _apply_brightness(
        self,
        node_id: int,
        endpoint_id: int,
        brightness: float,
        turn_on: bool = False
    ) -> Dict[str, Any]:
        """Apply brightness to a device. Returns result dict with 'error' key on failure."""
        level = int((brightness / 100) * 254)
        level = max(0, min(254, level))

        command = "MoveToLevelWithOnOff" if turn_on else "MoveToLevel"
        return self.client.device_command(
            node_id, endpoint_id, 8, command,
            {"level": level, "transitionTime": 0, "optionsMask": 0, "optionsOverride": 0}
        )

    def _apply_color_temp(
        self,
        node_id: int,
        endpoint_id: int,
        color_temp: str
    ) -> Dict[str, Any]:
        """Apply color temperature to a device. Returns result dict with 'error' key on failure."""
        temp_map = {
            "daylight": 153,
            "cool": 220,
            "neutral": 300,
            "warm": 400,
        }
        mireds = temp_map.get(color_temp, 300)
        return self.client.device_command(
            node_id, endpoint_id, 768, "MoveToColorTemperature",
            {"colorTemperatureMireds": mireds, "transitionTime": 0, "optionsMask": 0, "optionsOverride": 0}
        )

    def _execute_single_device(
        self,
        device_info: Dict[str, Any],
        action: str,
        brightness: Optional[float] = None,
        color_temp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a command on a single device."""
        node_id = device_info["node_id"]
        endpoint_id = device_info["endpoint_id"]
        device_name = device_info["name"]

        try:
            messages = []

            # Apply brightness/color_temp based on action and provided parameters
            # For "on" action, apply settings before turning on to avoid visual shift
            if action == "on":
                if brightness is not None:
                    result = self._apply_brightness(node_id, endpoint_id, brightness, turn_on=False)
                    if "error" in result:
                        return {"success": False, "device": device_name, "error": f"Failed to set brightness: {result['error']}"}
                    messages.append(f"brightness set to {brightness}%")

                if color_temp is not None:
                    result = self._apply_color_temp(node_id, endpoint_id, color_temp)
                    if "error" in result:
                        return {"success": False, "device": device_name, "error": f"Failed to set color temperature: {result['error']}"}
                    messages.append(f"color temperature set to {color_temp}")

                result = self.client.device_command(node_id, endpoint_id, 6, "On")
                if "error" in result:
                    return {"success": False, "device": device_name, "error": result["error"]}
                messages.append("turned on")

            elif action == "off":
                result = self.client.device_command(node_id, endpoint_id, 6, "Off")
                if "error" in result:
                    return {"success": False, "device": device_name, "error": result["error"]}
                messages.append("turned off")

            elif action == "toggle":
                result = self.client.device_command(node_id, endpoint_id, 6, "Toggle")
                if "error" in result:
                    return {"success": False, "device": device_name, "error": result["error"]}
                messages.append("toggled")

            elif action == "set_brightness" or brightness is not None:
                if brightness is None:
                    return {"success": False, "device": device_name, "error": "brightness parameter required"}

                if color_temp is not None:
                    result = self._apply_color_temp(node_id, endpoint_id, color_temp)
                    if "error" in result:
                        return {"success": False, "device": device_name, "error": f"Failed to set color temperature: {result['error']}"}
                    messages.append(f"color temperature set to {color_temp}")

                result = self._apply_brightness(node_id, endpoint_id, brightness, turn_on=True)
                if "error" in result:
                    return {"success": False, "device": device_name, "error": result["error"]}
                messages.append(f"brightness set to {brightness}%")

            elif action == "set_color_temp" or color_temp is not None:
                if color_temp is None:
                    return {"success": False, "device": device_name, "error": "color_temp parameter required"}

                if brightness is not None:
                    result = self._apply_brightness(node_id, endpoint_id, brightness, turn_on=True)
                    if "error" in result:
                        return {"success": False, "device": device_name, "error": f"Failed to set brightness: {result['error']}"}
                    messages.append(f"brightness set to {brightness}%")

                result = self._apply_color_temp(node_id, endpoint_id, color_temp)
                if "error" in result:
                    return {"success": False, "device": device_name, "error": result["error"]}
                messages.append(f"color temperature set to {color_temp}")

            else:
                return {"success": False, "device": device_name, "error": f"Unknown action: {action}"}

            return {"success": True, "device": device_name, "message": ", ".join(messages)}

        except Exception as e:
            return {"success": False, "device": device_name, "error": f"Error controlling light: {str(e)}"}

    def _aggregate_results(
        self,
        results: list,
        target: str,
        action: str,
        brightness: Optional[float] = None,
        color_temp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate results from multiple device operations."""
        total = len(results)
        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]

        success_count = len(successes)

        # Build message
        if success_count == total:
            # All succeeded
            if total == 1:
                return {"success": True, "message": f"{successes[0]['device']} {successes[0]['message']}"}
            else:
                return {"success": True, "message": f"{success_count} of {total} devices successfully completed action"}
        elif success_count > 0:
            # Partial success
            failure_details = ", ".join([f"{f['device']} ({f.get('error', 'unknown error')})" for f in failures])
            return {
                "success": True,
                "message": f"{success_count} of {total} devices successfully completed action. Failures: {failure_details}"
            }
        else:
            # All failed
            if total == 1:
                return {"success": False, "error": failures[0].get("error", "Unknown error")}
            else:
                failure_details = ", ".join([f"{f['device']} ({f.get('error', 'unknown error')})" for f in failures])
                return {"success": False, "error": f"All {total} devices failed: {failure_details}"}


class MatterListDevicesTool(Tool):
    """Tool to list available Matter devices."""

    def __init__(self, matter_host: str = "charmander.localdomain", matter_port: int = 5580):
        self.client = MatterClient(matter_host, matter_port) if WEBSOCKETS_AVAILABLE else None

    @property
    def name(self) -> str:
        return "list_matter_devices"

    @property
    def description(self) -> str:
        return "List all Matter devices that have been commissioned and their current state."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

    def execute(self) -> Dict[str, Any]:
        """List all commissioned Matter devices."""
        if not WEBSOCKETS_AVAILABLE:
            return {"success": False, "error": "websockets library not installed"}
        
        if not self.client:
            return {"success": False, "error": "Matter client not initialized"}

        try:
            result = self.client.get_nodes()
            
            if "error" in result:
                return {"success": False, "error": result["error"]}
            
            nodes = result.get("result", [])
            devices = []
            
            for node in nodes:
                node_id = node.get("node_id")
                attrs = node.get("attributes", {})
                
                # Extract useful info
                device_info = {
                    "node_id": node_id,
                    "available": node.get("available", False),
                    "vendor": attrs.get("0/40/1", "Unknown"),      # Basic Info cluster, VendorName
                    "product": attrs.get("0/40/14", "Unknown"),    # Basic Info cluster, ProductName  
                    "name": attrs.get("0/40/5", "Unknown"),        # Basic Info cluster, NodeLabel
                }
                
                # Check if it's a light (has OnOff cluster on endpoint 1)
                if "1/6/0" in attrs:
                    device_info["type"] = "light"
                    device_info["is_on"] = attrs.get("1/6/0", False)
                    device_info["brightness"] = attrs.get("1/8/0")  # Level cluster
                
                devices.append(device_info)
            
            return {
                "success": True,
                "count": len(devices),
                "devices": devices
            }

        except Exception as e:
            return {"success": False, "error": f"Error listing devices: {str(e)}"}


# Example usage and registration
if __name__ == "__main__":
    # Test the tool directly
    tool = MatterLightControlTool(
        matter_host="charmander.localdomain",
        matter_port=5580,
        device_aliases={
            "bathroom": {"node_id": 1, "endpoint_id": 1},
            "bedroom_globe": {"node_id": 2, "endpoint_id": 1},
            "bedroom_floor": {"node_id": 3, "endpoint_id": 1},
        },
        groups={
            "bedroom": ["bedroom_globe", "bedroom_floor"],
        }
    )

    # Test commands
    # Individual device control
    print(tool.execute(action="on", target="bathroom"))
    print(tool.execute(action="set_brightness", target="bedroom_globe", brightness=50))

    # Group control
    print(tool.execute(action="on", target="bedroom"))  # Controls both bedroom_globe and bedroom_floor
    print(tool.execute(action="set_brightness", target="bedroom", brightness=75))
    print(tool.execute(action="off", target="bedroom"))