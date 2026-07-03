# Benchmark For Embedding Task

example usage:

```shell
# debug mode
python bench_embed.py \
    --data-path ../data/lexical_embed.json \
    --debug

python bench_embed.py \
    --data-path ../data/lexical_embed.json \
    --qps 50 \
    --max-concurrency 50 \
    --num-requests 1000

python

```
