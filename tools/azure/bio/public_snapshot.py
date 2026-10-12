"""Package only the runtime and the named public benchmark, with an explicit source identity."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def snapshot(root, output):
    package = root / "src/aeviant_pipeline"
    source = sorted(package.rglob("*.py"))
    rows = [{"path": p.relative_to(package).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "size": p.stat().st_size} for p in source]
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    benchmark = Path("work-cycles/public-full-genome-benchmark-v1")
    names = [Path("pyproject.toml"), Path("README.md"),
        Path("src/aeviant_pipeline/rare_disease_profiles/exomiser-15.1.0-e0-standard-exome.yml"),
        Path("src/aeviant_pipeline/rare_disease_profiles/exomiser-15.1.0-e1-human-only-exome.yml"),
        benchmark / "campaign/runner/runner-manifest.json",
        benchmark / "campaign/runner/inputs/opaque-wgs-comphet-001.json",
        benchmark / "campaign/runner/inputs/opaque-wgs-comphet-001.vcf",
        Path("release/rare-disease-benchmark-v0.1.0/exomiser-pheval-production-campaign-v1/application.properties")]
    names += [p.relative_to(root) for p in source]
    # The CLI parser enumerates shipped generic profiles before dispatching any command.
    names.append(Path("src/aeviant_pipeline/research_profiles/mva_hackathon_2026.json"))
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file() or any(part.startswith(".") for part in name.parts):
            raise ValueError(f"not an approved regular input: {name}")
        if any(part in {"cases", "runtime-data", "evaluator"} for part in name.parts) or "avx" in str(name).lower():
            raise ValueError(f"protected input excluded: {name}")
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        for name in sorted(set(names)):
            archive.add(root / name, arcname="public-benchmark/" + name.as_posix(), recursive=False)
    receipt = {"adapter_source_manifest_sha256": identity, "source_file_count": len(rows),
               "snapshot_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
               "snapshot_bytes": output.stat().st_size, "files": sorted(str(n) for n in set(names)),
               "scope": "runtime source plus allowlisted public benchmark inputs; no cases or evaluator key"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {k: v for k, v in receipt.items() if k != "files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.pipeline.resolve(), args.output.resolve())))
