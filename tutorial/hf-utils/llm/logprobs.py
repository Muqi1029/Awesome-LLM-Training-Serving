import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda").eval()


def main():
    # step 1: generate content
    prompt = "What is the capital of France?"
    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    input_ids_len = len(inputs["input_ids"][0])
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_str = tokenizer.decode(
        generated_ids[0, input_ids_len:], skip_special_tokens=True
    )
    print(f"{generated_str=}")

    # step 2: forward to get logprobs
    output = model(generated_ids)
    logits = output.logits  # shape: [batch_size, seq_len, vocab_size]
    logprobs = F.log_softmax(logits, dim=-1)

    # step 3: computed prompt logprobs
    prompt_ids_index = generated_ids[:, 1:input_ids_len].unsqueeze(dim=-1)
    input_logprobs = logprobs[:, : input_ids_len - 1]
    prompt_logprobs = torch.gather(input_logprobs, dim=-1, index=prompt_ids_index)
    print(f"{prompt_logprobs=}")

    # step 4: computed output logprobs
    output_ids_index = generated_ids[:, input_ids_len:].unsqueeze(dim=-1)
    output_logprobs = logprobs[:, input_ids_len - 1 : -1]
    output_logprobs = torch.gather(output_logprobs, dim=-1, index=output_ids_index)
    print(f"{output_logprobs=}")


if __name__ == "__main__":
    main()
