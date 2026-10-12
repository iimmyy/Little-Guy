#!/bin/bash
# Sends what the nightly jobs need to storage: the engine fork's committed source (HEAD of .run/lcpp-src) for the
# compile VM, and the behavior tests, the persona and the nightly test script for the lg-tests job.
# Run after committing engine or test changes. Uncommitted engine edits are not included.
set -euo pipefail
cd "$(dirname "$0")/../.."
ST=lg$(grep -E '^LITTLEGUY_AZ_SUFFIX=' littleguy.env | cut -d= -f2)
TMP=$(mktemp -d)
git -C .run/lcpp-src archive --format=tar.gz -o "$TMP/lcpp-src.tar.gz" HEAD
az storage blob upload --account-name "$ST" -c build -n lcpp-src.tar.gz -f "$TMP/lcpp-src.tar.gz" --auth-mode login --overwrite -o none
tar czf "$TMP/tests.tar.gz" --exclude=__pycache__ tests/behavior persona/little_guy.private.txt persona/little_guy.example.txt
az storage blob upload --account-name "$ST" -c build -n tests.tar.gz -f "$TMP/tests.tar.gz" --auth-mode login --overwrite -o none
az storage blob upload --account-name "$ST" -c build -n nightly_tests.sh -f tools/azure/jobs/nightly_tests.sh --auth-mode login --overwrite -o none
echo "sent the tests, the persona and the nightly test script"
echo "sent engine $(git -C .run/lcpp-src log -1 --format='%h %s' | cut -c1-70) ($(du -h "$TMP/lcpp-src.tar.gz" | cut -f1))"
rm -rf "$TMP"
