"""Load and parse ~/.trussrc configuration for Baseten authentication."""

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TrussRemote:
    """Configuration for a single .trussrc remote/profile."""

    name: str
    api_key: str
    remote_url: str


class TrussrcLoader:
    """Load and parse ~/.trussrc configuration.

    The .trussrc file is an INI-style configuration file used by Baseten's
    truss CLI tool. It contains API keys and remote URLs for different
    Baseten environments/profiles.

    Example .trussrc format:
        [baseten]
        remote_url = https://app.baseten.co
        api_key = abc123...

        [staging]
        remote_url = https://app.staging.baseten.co
        api_key = xyz789...
    """

    DEFAULT_PATH = Path.home() / ".trussrc"

    def __init__(self, path: Optional[Path] = None):
        """Initialize the loader.

        Args:
            path: Path to .trussrc file. Defaults to ~/.trussrc.
        """
        self.path = path or self.DEFAULT_PATH
        self._config: Optional[ConfigParser] = None

    def load(self) -> bool:
        """Load the .trussrc file.

        Returns:
            True if file exists and was loaded successfully, False otherwise.
        """
        if not self.path.exists():
            return False
        self._config = ConfigParser()
        self._config.read(self.path)
        return True

    def get_profiles(self) -> List[str]:
        """Get list of available profile names.

        Returns:
            List of profile/section names from .trussrc.
        """
        if not self._config:
            return []
        return self._config.sections()

    def get_remote(self, profile: str) -> Optional[TrussRemote]:
        """Get configuration for a specific profile.

        Args:
            profile: The profile/section name (e.g., "baseten", "staging").

        Returns:
            TrussRemote with the profile's configuration, or None if not found.
        """
        if not self._config or profile not in self._config:
            return None
        section = self._config[profile]
        return TrussRemote(
            name=profile,
            api_key=section.get("api_key", ""),
            remote_url=section.get("remote_url", "https://app.baseten.co"),
        )

    def get_api_key(self, profile: str) -> Optional[str]:
        """Get the API key for a specific profile.

        Args:
            profile: The profile/section name.

        Returns:
            API key string, or None if profile not found.
        """
        remote = self.get_remote(profile)
        return remote.api_key if remote else None

    def get_inference_endpoint(self, profile: str) -> str:
        """Get the inference endpoint URL for a profile.

        Converts the app URL to the inference URL. For Baseten, this means
        converting https://app.baseten.co to https://inference.baseten.co/v1/chat/completions.

        Args:
            profile: The profile/section name.

        Returns:
            Inference endpoint URL, or empty string if profile not found.
        """
        remote = self.get_remote(profile)
        if not remote:
            return ""

        # Convert app URL to inference URL
        if "baseten.co" in remote.remote_url:
            # Handle different Baseten environments
            if "staging.baseten.co" in remote.remote_url:
                return "https://inference.staging.baseten.co/v1/chat/completions"
            elif "app.baseten.co" in remote.remote_url:
                return "https://inference.baseten.co/v1/chat/completions"
            else:
                # Custom Baseten environment - try to construct inference URL
                base = remote.remote_url.replace("app.", "inference.")
                return f"{base}/v1/chat/completions"

        # Non-Baseten remote, return as-is
        return remote.remote_url
