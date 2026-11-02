# Repository scope

This is an unofficial research implementation of Magical Athlete. The source and small examples are publicly available in this repository. Historical model weights, large experiment stores, and full research replay collections remain local. No project license has been selected.

The repository contains the Python implementation, a deterministic complete-game demonstration, a small learning example, portable executable tests, package/build instructions, derived evaluation-score tables, and explicit planned-attempt coverage. Standard-library scripts recompute the supplied statistics and coverage counts. The stopped final continuation has no after-model or score-improvement estimate.

Game roles, board coordinates, awards, and project rule-version identifiers are represented in the implementation. Source-attribution references are retained. Original third-party rulebooks, character-skill PDFs, scans, board images, artwork, and descriptive rule CSVs are not distributed here.

The repository also excludes private experiment manifests, historical trained weights, long replay collections, databases, local environments, and credentials. Those materials remain in the original local research workspace.

Two minimal test metadata files contain the implemented rule-version identifier and the curated case identifiers `T-001` through `T-033`. They contain no copied scenario descriptions and do not replace the original rules or audit evidence.

Runtime simulation uses the Python standard library. Setuptools is the build dependency; pytest is an optional test dependency. Third-party packages are installed separately and are not bundled in the repository.

The portable tests and deterministic demo validate behavior within their stated scope. They do not establish exhaustive correctness of every rule interaction, statistically reliable strategy improvement, or completion of the larger training program. The comparison report distinguishes training, source revalidation, diagnostic failure, independent evaluation, and the stopped continuation.

The public examples reproduce a complete Game, a small learning update, and statistics from the supplied score tables. They do not reconstruct the unavailable historical weights or independently authenticate the full private research archive. Publishing these derived tables does not make the original experiment databases and traces public.
