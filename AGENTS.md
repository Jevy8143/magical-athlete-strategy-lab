# Local source-package development guide

Use Python 3.12 or newer. The runtime uses only the standard library. Install
the package and optional test tools with `python -m pip install -e ".[test]"`.
Run the documented demo and the selected tests from the package root.

Preserve the four-player, four-Race game and its official cumulative score
ledger, including skill gains, losses, and Star points. Keep rules separate
from legal policy choices, and do not expose hidden future information.
Budget exhaustion is an incomplete diagnostic, never an invented finish.

Retain reproducible seeds and failure output. Do not overwrite existing demo
output. A passing demo or small test suite is not a strategy-strength result.
This package omits private models, experiments, and the full historical audit.

Read `docs/RELEASE_SCOPE.md` before redistribution. Third-party rulebooks and
artwork are not included, and no project license is added by this package.
Review the selected files and intended visibility before upload. Do not
publish or push anything without an explicit request.

Use `python tools/build_release.py` to audit the allowlist and
`python tools/build_release.py --build --name NEW-NAME` for a new local zip.
Existing release directories and evidence must be preserved.
