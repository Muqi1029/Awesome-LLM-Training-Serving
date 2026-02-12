import json
from argparse import ArgumentParser

from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True, help="")

    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument("--id-path")
    mutex_group.add_argument("--msg-path")
    mutex_group.add_argument("--payload-path")
    mutex_group.add_argument("--raw-text")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.id_path:
        input_ids = json.load(open(args.id_path, encoding="utf-8"))
        text = tokenizer.decode(input_ids)
        print(f"[DECODE RESULT] {text}")
    elif args.raw_text:
        input_ids = tokenizer.encode(args.raw_text)
        print(f"[ENCODE RESULT] {input_ids}")
    elif args.msg_path:
        with open(args.msg_path, "r", encoding="utf-8") as f:
            msg = json.load(f)
        text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        print(f"{text=}")
        input_ids = tokenizer.encode(text)
        print(f"[ENCODE RESULT] {input_ids}")
    elif args.payload_path:
        with open(args.msg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        msg = payload["messages"]
        text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        print(f"{text=}")
        input_ids = tokenizer.encode(text)
        print(f"[ENCODE RESULT] {input_ids}")
