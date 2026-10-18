"""Run the publisher's reviewed adapter against a matching local embeddings server."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from compare_jev_parity import compare


def run(args):
    spec = importlib.util.spec_from_file_location("jev_publisher_adapter", args.adapter)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    native = json.loads(args.native.read_text())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in native["cases"]:
        # The server adds BOS, while the NIF's fully rendered prompt contains it.
        prompt = case["rendered_prompt"].removeprefix("<bos>")
        request = Request(
            args.server.rstrip("/") + "/tokenize",
            data=json.dumps({"content": prompt, "add_special": True,
                             "parse_special": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=60) as response:
            token_ids = json.load(response)["tokens"]
        started = time.perf_counter_ns()
        reference = adapter.decide(args.server.rstrip("/"), args.head, case["input"],
                                   case["instructions"], case["options"])
        duration_us = (time.perf_counter_ns() - started) // 1000
        reference["duration_us"] = duration_us
        output_path = args.output_directory / (case["id"] + ".json")
        output_path.write_text(json.dumps(reference, indent=2))
        comparison = compare({"cases": [case]}, {case["id"]: reference})["cases"][0]
        reports.append({**comparison,
            "same_tokens": token_ids == case["token_ids"],
            "native_token_count": len(case["token_ids"]),
            "reference_token_count": len(token_ids),
            "reference_duration_us": duration_us,
        })
    report = {"native_metadata": native["metadata"], "cases": reports,
              "all_choices_match": all(case["same_choice"] for case in reports),
              "all_tokens_match": all(case["same_tokens"] for case in reports)}
    (args.output_directory / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["all_choices_match"] or not report["all_tokens_match"]:
        raise SystemExit("native/reference choice or token mismatch; inspect parity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Reviewed publisher jev_omni_gguf_decide.py from the pinned model revision")
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:18082")
    parser.add_argument("--output-directory", type=Path, required=True)
    run(parser.parse_args())
