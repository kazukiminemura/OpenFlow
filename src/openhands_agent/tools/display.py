from __future__ import annotations

import platform
import re
import subprocess

from .base import JsonDict, Tool, ToolResult


class DisplayTool(Tool):
    name = "display"
    description = "Control Windows display settings: list monitors, extend/clone displays, brightness, resolution, and orientation."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "mode", "brightness", "brightness_delta", "resolution", "resolution_delta", "orientation"],
                "description": "Display action to run.",
            },
            "mode": {
                "type": "string",
                "enum": ["extend", "clone", "internal", "external"],
                "description": "Projection mode for action=mode.",
            },
            "level": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Brightness percentage for action=brightness.",
            },
            "delta": {
                "type": "integer",
                "minimum": -100,
                "maximum": 100,
                "description": "Relative brightness change for action=brightness_delta.",
            },
            "direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Relative direction for action=resolution_delta.",
            },
            "width": {
                "type": "integer",
                "minimum": 320,
                "maximum": 10000,
                "description": "Screen width for action=resolution.",
            },
            "height": {
                "type": "integer",
                "minimum": 200,
                "maximum": 10000,
                "description": "Screen height for action=resolution.",
            },
            "orientation": {
                "type": "string",
                "enum": ["landscape", "portrait", "landscape_flipped", "portrait_flipped"],
                "description": "Screen orientation for action=orientation.",
            },
            "display": {
                "type": "string",
                "description": "Display target. Currently primary display is used for resolution/orientation.",
                "default": "primary",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum runtime before the command is stopped.",
                "default": 30,
                "minimum": 1,
                "maximum": 120,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def run(self, arguments: JsonDict) -> ToolResult:
        if platform.system() != "Windows":
            return ToolResult("Display control is only supported on Windows.", ok=False)

        action = str(arguments["action"])
        timeout = int(arguments.get("timeout_seconds", 30))

        if action == "list":
            return self._powershell(self._list_script(), timeout)
        if action == "mode":
            mode = str(arguments["mode"])
            return self._display_switch(mode, timeout)
        if action == "brightness":
            level = int(arguments["level"])
            return self._powershell(self._brightness_script(level), timeout)
        if action == "brightness_delta":
            delta = int(arguments["delta"])
            return self._powershell(self._brightness_delta_script(delta), timeout)
        if action == "resolution":
            width = int(arguments["width"])
            height = int(arguments["height"])
            return self._powershell(self._display_settings_script(width=width, height=height), timeout)
        if action == "resolution_delta":
            direction = str(arguments["direction"])
            return self._powershell(self._resolution_delta_script(direction), timeout)
        if action == "orientation":
            orientation = str(arguments["orientation"])
            return self._powershell(self._display_settings_script(orientation=orientation), timeout)

        return ToolResult(f"Unsupported display action: {action}", ok=False)

    def _display_switch(self, mode: str, timeout: int) -> ToolResult:
        switches = {
            "extend": "/extend",
            "clone": "/clone",
            "internal": "/internal",
            "external": "/external",
        }
        switch = switches.get(mode)
        if switch is None:
            return ToolResult(f"Unsupported display mode: {mode}", ok=False)

        completed = subprocess.run(
            ["DisplaySwitch.exe", switch],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return ToolResult(
            "\n".join(
                [
                    f"display_mode: {mode}",
                    f"exit_code: {completed.returncode}",
                    "stdout:",
                    completed.stdout.strip() or "<empty>",
                    "stderr:",
                    completed.stderr.strip() or "<empty>",
                ]
            ),
            ok=completed.returncode == 0,
        )

    def _powershell(self, script: str, timeout: int) -> ToolResult:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return ToolResult(
            "\n".join(
                [
                    f"exit_code: {completed.returncode}",
                    "stdout:",
                    completed.stdout.strip() or "<empty>",
                    "stderr:",
                    completed.stderr.strip() or "<empty>",
                ]
            ),
            ok=completed.returncode == 0,
        )

    def _list_script(self) -> str:
        return r"""
$monitors = Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID -ErrorAction SilentlyContinue
$brightness = Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBrightness -ErrorAction SilentlyContinue
$desktop = Get-CimInstance Win32_DesktopMonitor -ErrorAction SilentlyContinue
$result = [ordered]@{
  monitors = @($monitors | ForEach-Object {
    [ordered]@{
      instance = $_.InstanceName
      manufacturer = -join ($_.ManufacturerName | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ })
      name = -join ($_.UserFriendlyName | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ })
      serial = -join ($_.SerialNumberID | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ })
    }
  })
  brightness = @($brightness | ForEach-Object {
    [ordered]@{
      instance = $_.InstanceName
      current = $_.CurrentBrightness
      levels = $_.Level
    }
  })
  desktop_monitors = @($desktop | ForEach-Object {
    [ordered]@{
      name = $_.Name
      screen_width = $_.ScreenWidth
      screen_height = $_.ScreenHeight
      availability = $_.Availability
    }
  })
}
$result | ConvertTo-Json -Depth 6
"""

    def _brightness_script(self, level: int) -> str:
        if not 0 <= level <= 100:
            raise ValueError("brightness level must be between 0 and 100")
        return f"""
$methods = Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop
if (-not $methods) {{ throw "No brightness-capable display was found." }}
$methods | ForEach-Object {{
  Invoke-CimMethod -InputObject $_ -MethodName WmiSetBrightness -Arguments @{{ Timeout = 1; Brightness = {level} }} | Out-Null
}}
"brightness: {level}"
"""

    def _brightness_delta_script(self, delta: int) -> str:
        if not -100 <= delta <= 100:
            raise ValueError("brightness delta must be between -100 and 100")
        return f"""
$current = Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBrightness -ErrorAction Stop | Select-Object -First 1
if (-not $current) {{ throw "No brightness-capable display was found." }}
$target = [Math]::Max(0, [Math]::Min(100, [int]$current.CurrentBrightness + ({delta})))
$methods = Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop
$methods | ForEach-Object {{
  Invoke-CimMethod -InputObject $_ -MethodName WmiSetBrightness -Arguments @{{ Timeout = 1; Brightness = $target }} | Out-Null
}}
"brightness: $target"
"""

    def _display_settings_script(self, width: int | None = None, height: int | None = None, orientation: str | None = None) -> str:
        fields: list[str] = []
        if width is not None and height is not None:
            fields.append(f"$devMode.dmPelsWidth = {width}")
            fields.append(f"$devMode.dmPelsHeight = {height}")
            fields.append("$devMode.dmFields = $devMode.dmFields -bor 0x80000 -bor 0x100000")
        if orientation is not None:
            orientation_values = {
                "landscape": 0,
                "portrait": 1,
                "landscape_flipped": 2,
                "portrait_flipped": 3,
            }
            value = orientation_values[orientation]
            fields.append(f"$devMode.dmDisplayOrientation = {value}")
            fields.append("$devMode.dmFields = $devMode.dmFields -bor 0x80")
        if not fields:
            raise ValueError("width/height or orientation is required")

        field_script = "\n".join(fields)
        return rf"""
Add-Type @"
using System;
using System.Runtime.InteropServices;

public class DisplayNative {{
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
  public struct DEVMODE {{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
    public string dmDeviceName;
    public ushort dmSpecVersion;
    public ushort dmDriverVersion;
    public ushort dmSize;
    public ushort dmDriverExtra;
    public uint dmFields;
    public int dmPositionX;
    public int dmPositionY;
    public uint dmDisplayOrientation;
    public uint dmDisplayFixedOutput;
    public short dmColor;
    public short dmDuplex;
    public short dmYResolution;
    public short dmTTOption;
    public short dmCollate;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
    public string dmFormName;
    public ushort dmLogPixels;
    public uint dmBitsPerPel;
    public uint dmPelsWidth;
    public uint dmPelsHeight;
    public uint dmDisplayFlags;
    public uint dmDisplayFrequency;
    public uint dmICMMethod;
    public uint dmICMIntent;
    public uint dmMediaType;
    public uint dmDitherType;
    public uint dmReserved1;
    public uint dmReserved2;
    public uint dmPanningWidth;
    public uint dmPanningHeight;
  }}

  [DllImport("user32.dll", CharSet = CharSet.Auto)]
  public static extern int ChangeDisplaySettings(ref DEVMODE devMode, int flags);
}}
"@

$devMode = New-Object DisplayNative+DEVMODE
$devMode.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf([type][DisplayNative+DEVMODE])
$video = Get-CimInstance Win32_VideoController | Where-Object CurrentHorizontalResolution | Select-Object -First 1
if ($video) {{
  $devMode.dmPelsWidth = [uint32]$video.CurrentHorizontalResolution
  $devMode.dmPelsHeight = [uint32]$video.CurrentVerticalResolution
}}

{field_script}

$result = [DisplayNative]::ChangeDisplaySettings([ref]$devMode, 1)
if ($result -ne 0) {{ throw "ChangeDisplaySettings failed with code $result." }}
[ordered]@{{
  width = $devMode.dmPelsWidth
  height = $devMode.dmPelsHeight
  orientation = $devMode.dmDisplayOrientation
}} | ConvertTo-Json
"""

    def _resolution_delta_script(self, direction: str) -> str:
        if direction not in {"up", "down"}:
            raise ValueError("resolution direction must be up or down")
        compare = "-lt" if direction == "down" else "-gt"
        sort_desc = "$true" if direction == "down" else "$false"
        return rf"""
Add-Type @"
using System;
using System.Runtime.InteropServices;

public class DisplayNative {{
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
  public struct DEVMODE {{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
    public string dmDeviceName;
    public ushort dmSpecVersion;
    public ushort dmDriverVersion;
    public ushort dmSize;
    public ushort dmDriverExtra;
    public uint dmFields;
    public int dmPositionX;
    public int dmPositionY;
    public uint dmDisplayOrientation;
    public uint dmDisplayFixedOutput;
    public short dmColor;
    public short dmDuplex;
    public short dmYResolution;
    public short dmTTOption;
    public short dmCollate;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
    public string dmFormName;
    public ushort dmLogPixels;
    public uint dmBitsPerPel;
    public uint dmPelsWidth;
    public uint dmPelsHeight;
    public uint dmDisplayFlags;
    public uint dmDisplayFrequency;
    public uint dmICMMethod;
    public uint dmICMIntent;
    public uint dmMediaType;
    public uint dmDitherType;
    public uint dmReserved1;
    public uint dmReserved2;
    public uint dmPanningWidth;
    public uint dmPanningHeight;
  }}

  [DllImport("user32.dll", CharSet = CharSet.Auto)]
  public static extern int ChangeDisplaySettings(ref DEVMODE devMode, int flags);
}}
"@

$video = Get-CimInstance Win32_VideoController | Where-Object CurrentHorizontalResolution | Select-Object -First 1
if (-not $video) {{ throw "Current display resolution was not found." }}
$currentWidth = [int]$video.CurrentHorizontalResolution
$currentHeight = [int]$video.CurrentVerticalResolution
$currentArea = $currentWidth * $currentHeight

$candidates = @(
  [pscustomobject]@{{ Width = 3840; Height = 2160; Area = 3840 * 2160 }},
  [pscustomobject]@{{ Width = 3200; Height = 2000; Area = 3200 * 2000 }},
  [pscustomobject]@{{ Width = 2880; Height = 1800; Area = 2880 * 1800 }},
  [pscustomobject]@{{ Width = 2560; Height = 1600; Area = 2560 * 1600 }},
  [pscustomobject]@{{ Width = 2560; Height = 1440; Area = 2560 * 1440 }},
  [pscustomobject]@{{ Width = 2256; Height = 1504; Area = 2256 * 1504 }},
  [pscustomobject]@{{ Width = 2048; Height = 1536; Area = 2048 * 1536 }},
  [pscustomobject]@{{ Width = 1920; Height = 1200; Area = 1920 * 1200 }},
  [pscustomobject]@{{ Width = 1920; Height = 1080; Area = 1920 * 1080 }},
  [pscustomobject]@{{ Width = 1680; Height = 1050; Area = 1680 * 1050 }},
  [pscustomobject]@{{ Width = 1600; Height = 900; Area = 1600 * 900 }},
  [pscustomobject]@{{ Width = 1440; Height = 900; Area = 1440 * 900 }},
  [pscustomobject]@{{ Width = 1366; Height = 768; Area = 1366 * 768 }},
  [pscustomobject]@{{ Width = 1280; Height = 800; Area = 1280 * 800 }},
  [pscustomobject]@{{ Width = 1280; Height = 720; Area = 1280 * 720 }},
  [pscustomobject]@{{ Width = 1024; Height = 768; Area = 1024 * 768 }}
)
$modes = $candidates | Where-Object {{ $_.Area {compare} $currentArea }}

if (-not $modes) {{ throw "No {direction} resolution mode was found." }}
$orderedModes = $modes | Sort-Object Area -Descending:{sort_desc}
$target = $null
$lastResult = $null
foreach ($candidate in $orderedModes) {{
  $test = New-Object DisplayNative+DEVMODE
  $test.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf([type][DisplayNative+DEVMODE])
  $test.dmPelsWidth = [uint32]$candidate.Width
  $test.dmPelsHeight = [uint32]$candidate.Height
  $test.dmFields = 0x80000 -bor 0x100000
  $lastResult = [DisplayNative]::ChangeDisplaySettings([ref]$test, 2)
  if ($lastResult -eq 0) {{
    $target = $candidate
    break
  }}
}}
if (-not $target) {{ throw "No supported {direction} resolution mode was found. Last test result: $lastResult" }}

$current = New-Object DisplayNative+DEVMODE
$current.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf([type][DisplayNative+DEVMODE])
$current.dmPelsWidth = [uint32]$target.Width
$current.dmPelsHeight = [uint32]$target.Height
$current.dmFields = 0x80000 -bor 0x100000
$result = [DisplayNative]::ChangeDisplaySettings([ref]$current, 1)
if ($result -ne 0) {{ throw "ChangeDisplaySettings failed with code $result." }}
[ordered]@{{
  direction = "{direction}"
  width = $current.dmPelsWidth
  height = $current.dmPelsHeight
}} | ConvertTo-Json
"""


def parse_resolution(value: str) -> tuple[int, int] | None:
    match = re.search(r"(?P<width>\d{3,5})\s*[x×]\s*(?P<height>\d{3,5})", value, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group("width")), int(match.group("height"))
