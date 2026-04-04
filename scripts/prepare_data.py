import argparse
import json
from pathlib import Path

from pretraining_stack.data import synthetic_documents, write_packed_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Write deterministic packed token shards")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--eos-token-id", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    documents = synthetic_documents(
        args.documents, args.document_length, args.vocab_size, args.seed
    )
    manifest = write_packed_shards(
        documents,
        args.output,
        sequence_length=args.sequence_length,
        eos_token_id=args.eos_token_id,
        shard_count=args.shards,
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
