class PluginRegistryError(Exception):
    """Base for registry errors."""


class PluginUnreachableError(PluginRegistryError):
    """Plugin's bundle URL could not be reached (network error, DNS, timeout).

    Raised at refresh time; also surfaced when the executor tries to invoke
    a plugin whose registry entry is in `UNAVAILABLE` state.
    """

    def __init__(self, plugin_id: str, url: str, reason: str) -> None:
        super().__init__(f"Plugin '{plugin_id}' at {url} unreachable: {reason}")
        self.plugin_id = plugin_id
        self.url = url
        self.reason = reason


class PluginManifestInvalidError(PluginRegistryError):
    """Plugin returned a manifest that failed contract validation.

    Distinct from `PluginUnreachableError` so ops dashboards can separate
    "the network broke" from "the plugin is on the wrong contract version".
    """

    def __init__(self, plugin_id: str, url: str, reason: str) -> None:
        super().__init__(f"Plugin '{plugin_id}' at {url} returned invalid manifest: {reason}")
        self.plugin_id = plugin_id
        self.url = url
        self.reason = reason
