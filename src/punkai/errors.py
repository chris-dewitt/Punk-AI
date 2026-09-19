"""Error types. Every failure a user can cause should be one of these, so the
CLI and the server can turn them into a clean message instead of a traceback."""


class SmithError(Exception):
    """Base class for every expected failure."""


class ManifestError(SmithError):
    """A model manifest is missing fields, or asks for something unsafe."""


class VerificationError(SmithError):
    """Files on disk do not match the manifest."""


class LicenseError(SmithError):
    """The intended use is not permitted by the model's license."""


class PolicyError(SmithError):
    """A request was refused by a guard or serving policy."""


class BackendError(SmithError):
    """An inference backend failed or is not installed."""
