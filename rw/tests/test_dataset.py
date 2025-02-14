import sys
from pprint import pprint

import pytest
import yaml
from transformers import AutoTokenizer

pprint(sys.path)
sys.path.append("..")
from rw.code.prepare_data import get_dataloader


@pytest.mark.parametrize("config_path", ["config/hh.yaml"])
def test_dataloader(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    dataloader = get_dataloader(config, tokenizer)
    for batch in dataloader:
        print(batch["chosen_inputs"]["input_ids"].shape)
        print(batch["rejected_inputs"]["input_ids"].shape)
        break
