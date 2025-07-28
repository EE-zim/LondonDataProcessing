"""
Configuration constants and helper functions for LDP (London Data Processing).
"""
from pathlib import Path

# Supported log file extensions (used in file discovery and messaging)
SUPPORTED_EXTENSIONS = ('.txt', '.md', '.pdf', '.log', '.csv', '.json', '.xml')
_SUPPORTED_EXTENSIONS_STR = ', '.join(SUPPORTED_EXTENSIONS)

def is_supported_file(f: Path) -> bool:
    """Return True if path is a supported log file."""
    return f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS

def find_supported_files(root: Path) -> list[Path]:
    """Recursively find all supported log files under the given root path."""
    return [f for f in root.rglob('*') if is_supported_file(f)]
