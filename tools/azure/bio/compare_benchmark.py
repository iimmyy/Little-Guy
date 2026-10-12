"""Compare full frozen output and the predeclared legacy scientific fields separately."""
import argparse
import hashlib
import json
from pathlib import Path

RECORD_FIELDS = ("alt", "chrom", "end", "grouping_id", "inheritance_mode", "position", "rank", "ref", "reference_build", "score")
VARIANT_FIELDS = ("alt", "chrom", "position", "ref", "reference_build")
CASE_FIELDS = ("case_id", "ranked_gene_records", "ranked_genes", "ranked_variant_records", "ranked_variants")


def projection(result):
    cases = []
    for case in result["cases"]:
        if case["ranked_genes"] != [] or case["ranked_gene_records"] != []:
            raise ValueError("VARIANT-only reference has no gene ranking; new gene rankings differ")
        cases.append({"case_id": case["case_id"], "ranked_genes": case["ranked_genes"],
            "ranked_gene_records": case["ranked_gene_records"],
            "ranked_variant_records": [{key: row[key] for key in RECORD_FIELDS} for row in case["ranked_variant_records"]],
            "ranked_variants": [{key: row[key] for key in VARIANT_FIELDS} for row in case["ranked_variants"]]})
    return {"campaign_id": result["campaign_id"], "cases": cases}


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right)):
            child = path + "/" + key
            if key not in left:
                found.append({"path": child, "change": "added", "cloud": right[key]})
            elif key not in right:
                found.append({"path": child, "change": "missing", "mac": left[key]})
            else:
                found.extend(differences(left[key], right[key], child))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [change for index, (a, b) in enumerate(zip(left, right))
                for change in differences(a, b, path + "/" + str(index))]
    return [] if left == right else [{"path": path, "change": "changed", "mac": left, "cloud": right}]


def compare(cloud, mac):
    full = (cloud["normalized_results_sha256"] == mac["normalized_results_sha256"]
            and cloud["normalized_results"] == mac["normalized_results"]
            and cloud["runner_manifest_sha256"] == mac["runner_manifest_sha256"])
    error = None
    try:
        legacy = (projection(cloud["normalized_results"]) == projection(mac["normalized_results"])
                  and cloud["runner_manifest_sha256"] == mac["runner_manifest_sha256"])
    except (KeyError, TypeError, ValueError) as failure:
        legacy, error = False, str(failure)
    return {"frozen_results_match_mac": full, "legacy_scientific_fields_match_mac": legacy,
            "legacy_comparison_error": error, "legacy_case_fields": CASE_FIELDS,
            "legacy_ranked_record_fields": RECORD_FIELDS, "legacy_variant_fields": VARIANT_FIELDS,
            "comparison_rules": "exact values, original order, multiplicity, grouping, inheritance, ranks and scores; no tolerance",
            "normalized_output_differences": differences(mac["normalized_results"], cloud["normalized_results"]),
            "cloud_normalized_results_sha256": cloud["normalized_results_sha256"],
            "mac_normalized_results_sha256": mac["normalized_results_sha256"],
            "claim_authority": "DECLARED_NEW_SOURCE_COMPARISON_ONLY"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", type=Path)
    parser.add_argument("mac", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.cloud.read_text()), json.loads(args.mac.read_text()))
    result.update({"cloud_frozen_sha256": hashlib.file_digest(args.cloud.open("rb"), "sha256").hexdigest(),
                   "mac_frozen_sha256": hashlib.file_digest(args.mac.open("rb"), "sha256").hexdigest()})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("frozen_results_match_mac", "legacy_scientific_fields_match_mac")}))
    raise SystemExit(0 if result["frozen_results_match_mac"] else 1)
