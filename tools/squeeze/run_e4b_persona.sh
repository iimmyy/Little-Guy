#!/bin/bash
# E4B through the squeeze pipeline with persona drift as the main score: probe -> refine (4.66 GB) -> build -> score.
# Resumes where it stopped (finished probes are kept). Log: .run/logs/squeeze-e4b-persona.log
cd "$(dirname "$0")/../.."
CFG=tools/squeeze/gemma4-e4b-persona.json; Q="python3 tools/squeeze/squeeze.py $CFG"
{ $Q probe && $Q refine && $Q build && $Q score .run/weights/squeeze-e4b/work-persona/e4b-persona.gguf; echo "exit $?"; } >> .run/logs/squeeze-e4b-persona.log 2>&1
