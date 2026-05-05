# server.sh

shortcut to launch sglang server as soon as possible

the model_path should be `*/huggingface-model-name/commit-id`, like `/models/Qwen3-0.6B/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143`

usage example:
```shell
# default model is Qwen3-0.6B, default port is 8888
./server.sh

# change model
MODEL_PATH=/models/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 ./server.sh

# change port
PORT=9999 ./server.sh
```
