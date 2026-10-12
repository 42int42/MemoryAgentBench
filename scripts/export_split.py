"""
Export one split of the Hugging Face dataset, at a fixed revision, to a JSONL file
that main.py reads with --data_file.

Example:
    python scripts/export_split.py --split Conflict_Resolution \
        --revision 7ea066982b140a19337e17e60d45d4076e042faf --source factconsolidation_sh_6k --out exported_data/factconsolidation_sh_6k.jsonl
"""
import json
import os
from argparse import ArgumentParser

from datasets import load_dataset


def main():
    parser = ArgumentParser()
    parser.add_argument('--split', type=str, required=True,
                       help='Accurate_Retrieval, Test_Time_Learning, Long_Range_Understanding or Conflict_Resolution')
    parser.add_argument('--revision', type=str, required=True,
                       help='Commit of ai-hyz/MemoryAgentBench on Hugging Face')
    parser.add_argument('--source', type=str, default=None,
                       help='Keep only the samples of this sub_dataset (metadata.source)')
    parser.add_argument('--out', type=str, required=True,
                       help='The JSONL file to write')
    args = parser.parse_args()

    rows = load_dataset("ai-hyz/MemoryAgentBench", split=args.split, revision=args.revision)
    if args.source:
        rows = rows.filter(lambda sample: (sample.get("metadata") or {}).get("source", "") == args.source)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    count = {}
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in rows:
            # Two fields added for a harness, which main.py does not read: an id per sample
            # (its source and its place among that source's samples, from 1) and its
            # questions paired with their answers as conversation_creator.py pairs them
            source = (row.get("metadata") or {}).get("source", "")
            count[source] = count.get(source, 0) + 1
            questions = row["questions"] if isinstance(row["questions"], list) else [row["questions"]]
            answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
            if len(questions) > 1 and len(answers) > 1:
                qa = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
            else:
                qa = [{"question": questions[0] if questions else "", "answer": answers}]
            row = dict(row, id=f"{source}-{count[source]}", qa=qa)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} samples of {args.split} at {args.revision} to {args.out}")


if __name__ == '__main__':
    main()
