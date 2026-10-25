"""Connectivity and downstream-current reference audit, not circuit simulation."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def reachable(links, start, *, skip=None):
    adjacency = defaultdict(set)
    for edge in links:
        if edge["id"] != skip:
            adjacency[edge["a"]].add(edge["b"])
            adjacency[edge["b"]].add(edge["a"])
    seen = {start}
    pending = [start]
    while pending:
        for other in adjacency[pending.pop()] - seen:
            seen.add(other)
            pending.append(other)
    return seen


def audit(model):
    links = model["links"]
    ids = [edge["id"] for edge in links]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate link ID")
    loads = model["loads"]
    names = [load["name"] for load in loads]
    if len(names) != len(set(names)):
        raise ValueError("duplicate load name")
    errors = []
    domains = {domain: [e for e in links if e["domain"] == domain]
               for domain in ("positive", "return")}
    if sum(map(len, domains.values())) != len(links):
        raise ValueError("unknown power domain")
    positive_nodes = {e[k] for e in domains["positive"] for k in ("a", "b")}
    return_nodes = {e[k] for e in domains["return"] for k in ("a", "b")}
    if positive_nodes & return_nodes:
        errors.append("positive and return domains share nodes (modeled short)")
    for domain, edges in domains.items():
        source = model["source_nodes"][domain]
        connected = reachable(edges, source)
        for load in loads:
            if load[domain] not in connected:
                errors.append(f"{load['name']}: {domain} not connected to source")
        cycle_edges = [e["id"] for e in edges
                       if e["b"] in reachable(edges, e["a"], skip=e["id"])]
        if cycle_edges:
            errors.append(f"{domain} parallel paths/loop: {', '.join(cycle_edges)}")
    for node in model["explicitly_unconnected"]:
        if node in positive_nodes | return_nodes:
            errors.append(f"intentionally omitted connection is present: {node}")
    data = model["data_net"]
    if len(data["members"]) != len(loads) or set(data["members"]) != {l["data"] for l in loads}:
        errors.append("data membership does not match loads")
    cuts = []
    segments = []
    for edge in links:
        domain = edge["domain"]
        connected = reachable(domains[domain], model["source_nodes"][domain], skip=edge["id"])
        disconnected = [load for load in loads if load[domain] not in connected]
        if "expected_source_disconnects" in edge:
            expected = set(edge["expected_source_disconnects"])
            actual = {load["name"] for load in disconnected}
            matches = actual == expected
            cuts.append({"link": edge["id"], "matches_expected": matches,
                         "loads_disconnected_from_psu_positive": sorted(actual)})
            if not matches:
                errors.append(f"{edge['id']}: opening does not isolate expected loads from PSU positive")
        segments.append({"link": edge["id"], "domain": domain,
                         "loads_on_unique_path": [load["name"] for load in disconnected],
                         "rated_sum_reference_a": round(sum(l["rated_current_reference_a"] for l in disconnected), 3),
                         "stall_sum_reference_a": round(sum(l["stall_current_reference_a"] for l in disconnected), 3)})
    # The sums have a current-path meaning only for the intended radial graph.
    # They are neither measured currents nor fault-current/fuse calculations.
    return {"scope": "Ideal wire graph and published load sums only; external paths and device internals omitted",
            "topology_checks_passed": not errors, "errors": errors,
            "load_count": len(loads), "source_cuts": cuts,
            "segment_reference_sums_valid_for_radial_graph": not errors,
            "segment_reference_sums": segments,
            "all_parts_and_ratings_selected": all(e["selected_part"] is not None and e["current_rating_a"] is not None for e in links),
            "electrical_design_validated": False, "physical_harness_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "harness.json")
    parser.add_argument("--output", type=Path, default=ROOT / "harness-audit.json")
    args = parser.parse_args()
    result = audit(json.loads(args.model.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("topology_checks_passed", "errors", "load_count", "electrical_design_validated")}))
    return 0 if result["topology_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
