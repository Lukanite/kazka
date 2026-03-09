"""
Matter light control tools for the voice assistant.
Allows control of Matter-enabled lights via python-matter-server.
"""

from typing import Dict, Any, Optional
import asyncio
import json

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import your base Tool class - adjust path as needed
from core.tool_manager import Tool


class MatterClient:
    """Simple Matter server client for sending commands."""
    
    def __init__(self, host: str = "charmander.localdomain", port: int = 5580, timeout: float = 10.0):
        self.uri = f"ws://{host}:{port}/ws"
        self.timeout = timeout
        self._msg_id = 0
    
    def _next_msg_id(self) -> str:
        self._msg_id += 1
        return str(self._msg_id)
    
    async def _send_command(self, command: str, args: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a command to the Matter server and return the response."""
        try:
            async with websockets.connect(self.uri) as ws:
                # Consume server info message
                await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                
                message = {
                    "message_id": self._next_msg_id(),
                    "command": command,
                }
                if args:
                    message["args"] = args
                
                await ws.send(json.dumps(message))
                response = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                return json.loads(response)
        except asyncio.TimeoutError:
            return {"error": "Connection timeout"}
        except ConnectionRefusedError:
            return {"error": "Could not connect to Matter server"}
        except Exception as e:
            return {"error": str(e)}
    
    def send_command(self, command: str, args: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous wrapper for sending commands."""
        # Use asyncio.run() to create a new event loop - works in any thread
        return asyncio.run(self._send_command(command, args))
    
    def device_command(
        self,
        node_id: int,
        endpoint_id: int,
        cluster_id: int,
        command_name: str,
        payload: Optional[Dict] = None
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