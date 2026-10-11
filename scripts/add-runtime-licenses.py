"""Copy native runtime notices into a package directory before upload."""
from pathlib import Path
import sys
from runtime_licenses import copy_licenses

copy_licenses(Path(sys.argv[1]))
