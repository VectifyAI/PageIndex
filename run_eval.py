"""Evaluate translation quality using Vertex AI (COMET, BLEU, MetricX) + terminology recall."""

import argparse
import json
import os
import re

import pandas as pd
import vertexai
from google.oauth2 import service_account
from vertexai.evaluation import EvalTask

COMPOSITE_WEIGHTS = {
    "metricx_norm": 0.50,
    "comet": 0.25,
    "bleu": 0.15,
    "terminology_recall": 0.10,
}


def load_sources(batch_input_path):
    """Load English source texts keyed by custom_id from a batch input file."""
    sources = {}
    with open(batch_input_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sources[d["custom_id"]] = d["body"]["input"]
    return sources


def load_batch_output(batch_output_path):
    """Load model translations keyed by custom_id from an OpenAI batch output file."""
    translations = {}
    with open(batch_output_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            custom_id = d["custom_id"]
            resp = d.get("response", {})
            body = resp.get("body", {})
            # Handle /v1/responses format
            output_text = body.get("output_text", "")
            if not output_text:
                # Try output array format
                for item in body.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                output_text = content.get("text", "")
                                break
            translations[custom_id] = output_text
    return translations


def load_references(reference_path):
    """Load reference translations keyed by custom_id.

    Supports JSONL with 'custom_id' and 'translation'/'reference'/'hebrew' fields,
    or a simple two-column TSV (custom_id<tab>translation).
    """
    references = {}
    with open(reference_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{"):
            # JSONL format
            for line in f:
                d = json.loads(line)
                cid = d.get("custom_id", d.get("id", ""))
                ref = d.get("translation", d.get("reference", d.get("hebrew", "")))
                references[cid] = ref
        elif "\t" in first_line:
            # TSV format
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    references[parts[0]] = parts[1]
        else:
            # Try CSV
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("custom_id", row.get("id", ""))
                ref = row.get("translation", row.get("reference", row.get("hebrew", "")))
                references[cid] = ref
    return references


def load_glossary(glossary_path):
    with open(glossary_path, encoding="utf-8") as f:
        return json.load(f)


def compute_terminology_recall(source, response, glossary):
    """Compute recall of glossary terms: how many expected Hebrew terms appear in the response."""
    source_lower = source.lower()
    # Sort glossary terms by length (longest first) to avoid partial matches
    sorted_terms = sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True)

    expected = 0
    matched = 0
    matched_terms_list = []

    for en_term, he_term in sorted_terms:
        if en_term.lower() in source_lower:
            expected += 1
            if he_term in response:
                matched += 1
                matched_terms_list.append(en_term)

    recall = matched / expected if expected > 0 else 1.0
    return recall, matched, expected


def normalize_metricx(score):
    """Normalize MetricX from 0-25 (lower=better) to 0-1 (higher=better)."""
    return max(0.0, 1.0 - score / 25.0)


def composite_score(row):
    """Compute weighted composite score from normalized metrics."""
    return sum(row[metric] * weight for metric, weight in COMPOSITE_WEIGHTS.items())


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation quality")
    parser.add_argument("--batch-output", required=True, help="Batch output JSONL from OpenAI")
    parser.add_argument("--reference", required=True, help="Reference translations file")
    parser.add_argument("--glossary", default="glossary.json", help="Glossary JSON file")
    parser.add_argument("--source-batch", required=True, help="Original batch input JSONL (for source texts)")
    parser.add_argument("--gcp-project", default=os.getenv("GCP_PROJECT"), help="GCP project ID")
    parser.add_argument("--gcp-location", default=os.getenv("GCP_LOCATION", "us-central1"), help="GCP location")
    parser.add_argument("--gcp-credentials", default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), help="Service account JSON path")
    parser.add_argument("--output", default="eval_results.csv", help="Output CSV path")
    args = parser.parse_args()

    if not args.gcp_project:
        parser.error("--gcp-project or GCP_PROJECT env var is required")

    # Load data
    print("Loading data...")
    sources = load_sources(args.source_batch)
    translations = load_batch_output(args.batch_output)
    references = load_references(args.reference)
    glossary = load_glossary(args.glossary)

    # Align by custom_id
    common_ids = sorted(set(sources) & set(translations) & set(references))
    print(f"Aligned {len(common_ids)} entries (sources={len(sources)}, translations={len(translations)}, references={len(references)})")

    if not common_ids:
        print("ERROR: No matching custom_ids found across all three files.")
        return

    df = pd.DataFrame({
        "custom_id": common_ids,
        "source": [sources[cid] for cid in common_ids],
        "response": [translations[cid] for cid in common_ids],
        "reference": [references[cid] for cid in common_ids],
    })

    # Vertex AI evaluation
    print("Initializing Vertex AI...")
    credentials = None
    if args.gcp_credentials:
        credentials = service_account.Credentials.from_service_account_file(args.gcp_credentials)
    vertexai.init(project=args.gcp_project, location=args.gcp_location, credentials=credentials)

    print("Running Vertex AI evaluation (COMET, BLEU, MetricX)...")
    # MetricX auto-selects METRICX_24_SRC_REF mode when "reference" column is present
    eval_df = df[["source", "response", "reference"]].copy()
    eval_task = EvalTask(
        dataset=eval_df,
        metrics=["comet", "metricx", "bleu"],
    )
    result = eval_task.evaluate()
    print("MetricX mode: SRC_REF (reference-based, auto-detected from reference column)")

    # Merge per-row scores
    metrics_df = result.metrics_table
    for col in ["comet", "metricx", "bleu"]:
        if col in metrics_df.columns:
            df[col] = metrics_df[col].values
        elif f"{col}/score" in metrics_df.columns:
            df[col] = metrics_df[f"{col}/score"].values

    # Terminology recall
    print("Computing terminology recall...")
    term_results = df.apply(
        lambda row: compute_terminology_recall(row["source"], row["response"], glossary),
        axis=1,
        result_type="expand",
    )
    df["terminology_recall"] = term_results[0]
    df["matched_terms"] = term_results[1]
    df["expected_terms"] = term_results[2]

    # Normalize and composite
    df["metricx_norm"] = df["metricx"].apply(normalize_metricx)
    df["composite_score"] = df.apply(composite_score, axis=1)

    # Output
    output_cols = [
        "custom_id", "source", "response", "reference",
        "comet", "bleu", "metricx", "metricx_norm",
        "terminology_recall", "matched_terms", "expected_terms",
        "composite_score",
    ]
    df[output_cols].to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n=== Summary Metrics ===")
    print(f"  COMET (0-1, higher=better):           {df['comet'].mean():.4f}")
    print(f"  BLEU (0-1, higher=better):             {df['bleu'].mean():.4f}")
    print(f"  MetricX (0-25, lower=better):          {df['metricx'].mean():.4f}")
    print(f"  MetricX normalized (0-1, higher=better):{df['metricx_norm'].mean():.4f}")
    print(f"  Terminology recall (0-1):              {df['terminology_recall'].mean():.4f}")
    print(f"  Composite score (0-1):                 {df['composite_score'].mean():.4f}")
    print(f"\n  Composite weights: {COMPOSITE_WEIGHTS}")


if __name__ == "__main__":
    main()
