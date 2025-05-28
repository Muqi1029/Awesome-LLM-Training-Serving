

preprocess_data_py=../Megatron-LM/tools/preprocess_data.py
vocab_file=../../../checkpoints/megatron_bert_345m_v0.1_uncased/bert-large-uncased-vocab.txt
corpus_file=my-corpus.jsonl
output_prefix=my_bert

python $preprocess_data_py \
       --input $corpus_file \
       --output-prefix $output_prefix \
       --vocab-file $vocab_file \
       --tokenizer-type BertWordPieceLowerCase \
       --workers 2 \
       --split-sentences

