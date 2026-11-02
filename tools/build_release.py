"""Build a small local source handoff from an explicit allowlist; never publish.

Run without --build to audit the current inputs. The source workspace and its
private data are never rewritten. Existing output directories are retained.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tools/release_allowlist.json"
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
FORBIDDEN_PARTS = {".git", ".cache", "artifacts", "envs", ".venv", "venv", "__pycache__", "releases"}
FORBIDDEN_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".sqlite", ".sqlite3", ".db", ".pyc", ".pem", ".key", ".p12", ".pfx"}
PRIVATE_PATTERNS = {
    "absolute Windows path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "personal home path": re.compile(r"(?<![A-Za-z0-9:])/(?:Users|home)/[^\s\"']+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,})"),
    "cloud credential token": re.compile(r"\b(?:(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[0-9A-Za-z_-]{35}|github_pat_[0-9A-Za-z_]{30,}|xox[baprs]-[0-9A-Za-z-]{20,})\b"),
}
PORTABLE_DEVELOPER_GUIDE = """# Local source-package development guide

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
"""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def relative_name(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("nonportable release path")
    if any(part in FORBIDDEN_PARTS for part in path.parts) or path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        raise ValueError("excluded release category: " + value)
    return path.as_posix()


def read_source(name: str) -> bytes:
    name = relative_name(name)
    path = ROOT / name
    if path.resolve() != path or not path.is_file():
        raise ValueError("missing file or symbolic link: " + name)
    for item in (path, *path.parents):
        if not item.is_relative_to(ROOT):
            break
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError("release inputs must not use links/reparse points: " + name)
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("release file exceeds the small-package limit: " + name)
    return raw


def audit_text(name: str, raw: bytes) -> None:
    text = raw.decode("utf-8-sig")
    for label, pattern in PRIVATE_PATTERNS.items():
        found = pattern.search(text)
        if found:
            # Report the location, never the potentially sensitive value.
            line = text.count("\n", 0, found.start()) + 1
            raise ValueError(f"{label} requires review: {name}:{line}")
    aliases = {Path.home().name, os.environ.get("USERNAME", "")}
    for alias in aliases:
        if len(alias) >= 4 and alias.lower() not in {"root", "user", "admin", "administrator", "runner"} and alias in text:
            raise ValueError("local account name requires review: " + name)


def implemented_rule_version() -> str:
    tree = ast.parse(read_source("src/magical_athlete/engine.py").decode("utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "RULE_VERSION" for target in node.targets):
            value = ast.literal_eval(node.value)
            if type(value) is str and value:
                return value
    raise ValueError("implemented rule version is missing")


def release_files() -> tuple[dict[str, bytes], dict[str, str]]:
    configuration = json.loads(read_source("tools/release_allowlist.json"))
    if configuration["schema"] != 1:
        raise ValueError("unknown release allowlist schema")
    package = ROOT / relative_name(configuration["python_package"])
    names = [*configuration["root_files"], *configuration["tests"],
             *(path.relative_to(ROOT).as_posix() for path in sorted(package.glob("*.py")))]
    if len(set(names)) != len(names):
        raise ValueError("duplicate release input")
    original = {name: read_source(name) for name in names}
    source_hashes = {name: digest(raw) for name, raw in original.items()}
    output = dict(original)
    output["AGENTS.md"] = PORTABLE_DEVELOPER_GUIDE.encode("utf-8")
    # The workspace ignores its private manifest/CSV. Only generated minimal
    # replacements are made visible to Git in the extracted source handoff.
    exceptions = b"\n# Generated minimal release test metadata, not the private workspace originals.\n!docs/rule_manifest.json\n!docs/test_vectors_curated.csv\n"
    if exceptions not in output[".gitignore"]:
        output[".gitignore"] += exceptions
    output["docs/rule_manifest.json"] = canonical({
        "canonical_version": implemented_rule_version(),
        "scope": "runtime and selected release tests only",
        "historical_audit_documents_included": False,
    })
    output["docs/test_vectors_curated.csv"] = ("test_id\n" + "".join(f"T-{number:03d}\n" for number in range(1, 34))).encode("utf-8-sig")
    for name, raw in output.items():
        relative_name(name)
        audit_text(name, raw)
        if name.endswith(".md"):
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", raw.decode("utf-8-sig")):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                linked = posixpath.normpath(str(PurePosixPath(name).parent / target))
                if linked not in output and linked != "RELEASE_MANIFEST.json":
                    raise ValueError("unbundled local documentation link: " + name + " -> " + linked)
    if sum(map(len, output.values())) > MAX_TOTAL_BYTES:
        raise ValueError("release exceeds the small-package size limit")
    manifest = {
        "schema": 1,
        "scope": "local source handoff; no upload or public release performed",
        "license_selected": False,
        "third_party_PDFs_scans_or_artwork_included": False,
        "game_derived_names_and_rule_data_present": True,
        "models_training_databases_and_historical_replays_included": False,
        "runtime_dependencies": "Python standard library",
        "rule_version": implemented_rule_version(),
        "tests": configuration["tests"],
        "generated_minimal_metadata": ["AGENTS.md", "docs/rule_manifest.json", "docs/test_vectors_curated.csv"],
        "files": {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in sorted(output.items())},
    }
    output["RELEASE_MANIFEST.json"] = canonical(manifest)
    return output, source_hashes


def verify_stage(stage: Path, expected: dict[str, bytes]) -> None:
    actual = {path.relative_to(stage).as_posix(): path.read_bytes() for path in stage.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("staging bytes or membership differ from the audited allowlist")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="create a new local staging directory and zip")
    parser.add_argument("--name", default="magical-athlete-local-20261102")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.name) or args.name.endswith("."):
        raise ValueError("portable local release name required")
    output, source_hashes = release_files()
    result = {"audited": True, "built": False, "files": len(output),
              "uncompressed_bytes": sum(map(len, output.values())), "network_or_publication": False}
    if not args.build:
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    destination = ROOT / "releases"
    destination.mkdir(exist_ok=True)
    if destination.resolve() != destination or getattr(destination.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("release output must not use links/reparse points")
    stage, archive = destination / args.name, destination / (args.name + ".zip")
    if stage.exists() or archive.exists():
        raise ValueError("existing release retained; choose a new name")
    stage.mkdir()
    for name, raw in sorted(output.items()):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    verify_stage(stage, output)
    if any(digest(read_source(name)) != pin for name, pin in source_hashes.items()):
        raise ValueError("source changed during packaging; incomplete output retained")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, raw in sorted(output.items()):
            info = zipfile.ZipInfo(args.name + "/" + name, (2026, 11, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, raw, compresslevel=9)
    with zipfile.ZipFile(archive) as bundle:
        archived = {name.removeprefix(args.name + "/"): bundle.read(name) for name in bundle.namelist()}
        if archived != output or bundle.testzip() is not None:
            raise ValueError("zip content differs from staging")
    result.update(built=True, staging=stage.relative_to(ROOT).as_posix(), zip=archive.relative_to(ROOT).as_posix(),
                  zip_bytes=archive.stat().st_size, zip_sha256=digest(archive.read_bytes()))
    receipt = destination / (args.name + "-build.json")
    with receipt.open("xb") as stream:
        stream.write(canonical(result))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print("Release audit failed: " + str(error), file=sys.stderr)
        raise SystemExit(2)
