"""Compare real native reports with the publisher's per-question JSON outputs."""
import argparse
import json
import math
from pathlib import Path


def compare(native, references):
    cases = {case["id"]: case for case in native["cases"]}
    reports = []
    for case_id, reference in references.items():
        case = cases[case_id]
        probabilities = [reference["probabilities"][label] for label in case["options"]]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   and 0 <= value <= 1 for value in probabilities):
            raise ValueError("invalid reference distribution")
        if len(case["distribution"]) != len(probabilities):
            raise ValueError("distribution dimension mismatch")
        reference_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        reports.append({"id": case_id,
                        "same_choice": case["index"] == reference_index,
                        "max_absolute_probability_drift": max(
                            abs(a - b) for a, b in zip(case["distribution"], probabilities)),
                        "native_duration_us": case["duration_us"]})
    return {"cases": reports, "all_choices_match": all(case["same_choice"] for case in reports)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("--reference", action="append", required=True,
                        help="CASE_ID=publisher-result.json; repeat for each case")
    args = parser.parse_args()
    references = {}
    for item in args.reference:
        case_id, filename = item.split("=", 1)
        references[case_id] = json.loads(Path(filename).read_text())
    print(json.dumps(compare(json.loads(args.native.read_text()), references), indent=2))
