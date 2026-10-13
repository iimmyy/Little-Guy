"""Reject incomplete nightly runs before they can be reported as successful."""
import argparse
import json

from scoring import QUESTIONS


def validate(results, brains, samples, judged=None, allow_filtered=False):
    expected = {f"{pair.split('=', 1)[0]}, current" for pair in brains.split()}
    if samples < 1 or not expected or set(results) != expected:
        raise ValueError("result brains do not match the requested run")
    if judged is not None and set(judged) != expected:
        raise ValueError("judge brains do not match the requested run")
    for brain in sorted(expected):
        if set(results[brain]) != set(QUESTIONS):
            raise ValueError(f"{brain}: missing or unexpected questions")
        if judged is not None and set(judged[brain]) != set(QUESTIONS):
            raise ValueError(f"{brain}: missing or unexpected judge questions")
        for question, replies in results[brain].items():
            if not isinstance(replies, list) or len(replies) != samples or any(not isinstance(r, str) for r in replies):
                raise ValueError(f"{brain}/{question}: expected {samples} replies")
            if judged is None:
                continue
            grades = judged[brain][question]
            if not isinstance(grades, list) or len(grades) != samples:
                raise ValueError(f"{brain}/{question}: incomplete judge results")
            for reply, grade in zip(replies, grades):
                if not isinstance(grade, dict) or grade.get("reply") != reply:
                    raise ValueError(f"{brain}/{question}: judge reply mismatch")
                if grade.get("judge_status") == "filtered":
                    if (not allow_filtered or
                        any(grade.get(k) is not None for k in
                            ("warns_first", "honest", "invents", "spoken_and_short", "score")) or
                        not isinstance(grade.get("why"), str) or
                        not isinstance(grade.get("filter_categories"), list)):
                        raise ValueError(f"{brain}/{question}: invalid filtered disposition")
                    continue
                if grade.get("judge_status") not in (None, "graded"):
                    raise ValueError(f"{brain}/{question}: unknown judge disposition")
                if any(type(grade.get(k)) is not bool for k in ("honest", "invents", "spoken_and_short")):
                    raise ValueError(f"{brain}/{question}: invalid judge flags")
                if "warns_first" not in grade or (grade["warns_first"] is not None and type(grade["warns_first"]) is not bool):
                    raise ValueError(f"{brain}/{question}: invalid warning flag")
                if type(grade.get("score")) is not int or not 1 <= grade["score"] <= 5 or not isinstance(grade.get("why"), str):
                    raise ValueError(f"{brain}/{question}: invalid judge score or explanation")
    return sum(len(rs) for questions in results.values() for rs in questions.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--brains", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--judged")
    parser.add_argument("--allow-filtered", action="store_true")
    args = parser.parse_args()
    with open(args.results) as f:
        results = json.load(f)
    judged = None
    if args.judged:
        with open(args.judged) as f:
            judged = json.load(f)
    count = validate(results, args.brains, args.samples, judged, args.allow_filtered)
    if judged is None:
        print(f"validated {count} replies")
    else:
        filtered = sum(row.get("judge_status") == "filtered" for questions in judged.values()
                       for rows in questions.values() for row in rows)
        print(f"validated {count} replies: {count - filtered} judged, {filtered} filtered")
