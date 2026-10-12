"""Index approved synthetic behavior-test results for regression review."""

import argparse
import base64
import json
import os
import subprocess
import urllib.error
import urllib.request


INDEX = "behavior-results"
API = "2024-07-01"


def request(method, path, body=None):
    token = subprocess.run(
        ["az", "account", "get-access-token", "--resource", "https://search.azure.com",
         "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()
    endpoint = os.environ["LG_SEARCH_ENDPOINT"].rstrip("/")
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{endpoint}/{path}?api-version={API}", data=payload, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = response.read()
            return json.loads(data) if data else None
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Search HTTP {error.code}: {error.read().decode()[:500]}") from error


def field(name, type_, **kwargs):
    return {"name": name, "type": type_, **kwargs}


def ensure_index():
    body = {"name": INDEX, "fields": [
        field("id", "Edm.String", key=True, filterable=True),
        field("day", "Edm.String", filterable=True, facetable=True),
        field("run_id", "Edm.String", filterable=True),
        field("brain", "Edm.String", filterable=True, facetable=True),
        field("question", "Edm.String", filterable=True, facetable=True),
        field("sample", "Edm.Int32", filterable=True, sortable=True),
        field("score", "Edm.Int32", filterable=True, sortable=True),
        field("warns_first", "Edm.Boolean", filterable=True),
        field("honest", "Edm.Boolean", filterable=True),
        field("invents", "Edm.Boolean", filterable=True),
        field("spoken_and_short", "Edm.Boolean", filterable=True),
        field("reply", "Edm.String", searchable=True),
        field("why", "Edm.String", searchable=True),
    ]}
    return request("PUT", f"indexes/{INDEX}", body)


def documents(path, run_id, day):
    results = json.load(open(path))
    for brain, questions in results.items():
        for question, rows in questions.items():
            for sample, row in enumerate(rows):
                key = base64.urlsafe_b64encode(
                    f"{run_id}:{brain}:{question}:{sample}".encode()
                ).decode()
                yield {"@search.action": "mergeOrUpload", "id": key,
                       "day": day, "run_id": run_id, "brain": brain, "question": question,
                       "sample": sample, "score": int(row["score"]),
                       "warns_first": row["warns_first"], "honest": bool(row["honest"]),
                       "invents": bool(row["invents"]),
                       "spoken_and_short": bool(row["spoken_and_short"]),
                       "reply": row["reply"], "why": row["why"]}


def upload(path, run_id, day):
    count = 0
    batch = []
    for document in documents(path, run_id, day):
        batch.append(document)
        if len(batch) == 500:
            count += send_batch(batch)
            batch.clear()
    if batch:
        count += send_batch(batch)
    print(json.dumps({"indexed": count, "run_id": run_id, "day": day}))


def send_batch(batch):
    response = request("POST", f"indexes/{INDEX}/docs/index", {"value": batch})
    failed = [item for item in response["value"] if not item["status"]]
    if failed:
        raise RuntimeError(f"Search rejected {len(failed)} of {len(batch)} documents: {failed[:2]}")
    return len(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["ensure-index", "upload", "count"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("day", nargs="?")
    args = parser.parse_args()
    if args.action == "ensure-index":
        ensure_index()
        print(json.dumps({"index": INDEX, "status": "ready"}))
    elif args.action == "count":
        result = request("POST", f"indexes/{INDEX}/docs/search",
                         {"search": "*", "count": True, "top": 0})
        print(json.dumps({"index": INDEX, "documents": result["@odata.count"]}))
    else:
        if not all((args.path, args.run_id, args.day)):
            parser.error("upload requires path, run_id and day")
        upload(args.path, args.run_id, args.day)
