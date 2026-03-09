"""
Audio device manager for handling input/output device detection and configuration.
"""

import sounddevice as sd
from typing import Optional, Tuple
from core.config import config


class AudioDeviceManager:
    """Manages audio input/output device detection and configuration."""

    @staticmethod
    def get_device_id(substring: str, kind: str = 'input') -> Optional[int]:
        """
        Finds audio device ID by name.

        Args:
            substring: Substring to search for in device names
            kind: 'input' or 'output'

        Returns:
            Device ID or None if not found
        """
        print(f"🔎 Scanning for {kind} device: '{substring}'...")
        devices = sd.query_devices()

        for i, dev in enumerate(devices):
            if kind == 'input':
                channels = dev['max_input_channels']
            else:
                channels = dev['max_output_channels']

            if channels > 0 and substring.lower() in dev['name'].lower():
                print(f"   ✅ Found: {dev['name']} (ID: {i})")
                return i

        print(f"   ❌ Error: No {kind} found matching '{substring}'. Using Default.")
        return None

    @staticmethod
    def get_alsa_card_index(substring: str) -> str:
        """
        Reads /proc/asound/cards to find the ALSA Card ID for a device name.

        Args:
            substring: Substring to search for in ALSA card names

        Returns:
            ALSA card number as string
        """
        print(f"🔎 Scanning ALSA cards for: '{substring}'...")
        try:
            with open("/proc/asound/cards", "r") as f:
                for line in f:
                    # Line format example:
                    # " 1 [Olympus2       ]: USB-Audio - FiiO E10K Olympus 2"
                    if substring.lower() in line.lower():
                        # Get the number at the start of the line
                        card_id = line.split('[')[0].strip()
                        print(f"   ✅ Found Card {card_id}: {line.strip()}")
                        return card_id
        except FileNotFoundError:
            print("   ❌ /proc/asound/cards not found (Are you on Linux?)")

        print(f"   ❌ Error: No card found matching '{substring}'. Defaulting to Card 0.")
        return "0"

    def get_input_device_id(self) -> Optional[int]:
        """Get the configured input device ID."""
        return self.get_device_id(config.audio_devices.input_device_name, 'input')

    def get_output_device_id(self) -> Optional[int]:
        """Get the configured output device ID."""
        return self.get_device_id(config.audio_devices.output_device_name, 'output')

    def validate_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Validate and return device IDs for configured input/output devices.

        Returns:
            Tuple of (input_device_id, output_device_id)
        """
        mic_id = self.get_input_device_id()
        spk_id = self.get_output_device_id()

        # Log device information
        devices = sd.query_devices()
        print(f"\n🔊 Available Audio Devices:")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  INPUT {i}: {dev['name']}")
            if dev['max_output_channels'] > 0:
                print(f"  OUTPUT {i}: {dev['name']}")

        return mic_id, spk_id