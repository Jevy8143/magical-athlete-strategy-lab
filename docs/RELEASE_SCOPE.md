# Local source-package scope

This is an unofficial simulator and a local handoff candidate. It has not been published, uploaded, or pushed to a remote service. No project license has been selected or added.

The package contains the Python implementation, a small deterministic demonstration, selected executable tests, package/build instructions, small derived evaluation-score tables, and explicit planned-attempt coverage. Standard-library scripts recompute the supplied statistics and coverage counts. The stopped final continuation has no after-model or score-improvement estimate. Game roles, board coordinates, awards, and project rule-version identifiers are represented in the implementation. Source-attribution references in the code are retained; they do not grant redistribution permission.

The package does not contain the official rulebook, character-skill PDFs, scans, board images, long rules/audit documents, the original descriptive CSVs, private experiment manifests, trained weights, replay collections, databases, local environments, or credentials. Those materials remain in the original local workspace.

Two small test metadata files are generated during packaging: a JSON object containing the implemented rule-version identifier, and an index containing only the 33 case identifiers `T-001` through `T-033`. They contain no copied scenario descriptions and are not a replacement for the original rules or audit evidence.

The runtime uses the Python standard library. Setuptools is the build dependency; pytest is an optional test dependency. Third-party packages are obtained separately and are not bundled in this source archive.

This packaging work does not make a legal-clearance determination or choose a license. The source references identify where the implementation's game information came from; the referenced documents and artwork are not distributed in this package. The user should review the selected files, license choice, and intended visibility before any upload.

The release tests and deterministic demo validate the packaged behavior within their stated scope. They do not establish strategy improvement, completion of the larger training program, or exhaustive coverage of every interaction. The comparison report distinguishes historical training, source revalidation, diagnostic failure, independent evaluation, and the stopped continuation. Large raw research archives remain local; publishing small derived tables does not include or reproduce those original archives.
