# Transformer

```text
.
├── data/
|   ├── original/
|   |   ├── test.parquet
|   |   ├── train-wikitext103-1.parquet
|   |   ├── train-wikitext103-2.parquet
|   |   ├── train-wikitext103-ALL.parquet     // made using huggingface dataset export
|   |   ├── train-wikitext2.parquet
|   |   └── validation.parquet
|   ├── bpe.py
|   ├── dataset_creator.py
|   ├── dataset_creator_hf.py
|   ├── special_chars.py
|   └── wikitext.py
├── model/
|   ├── bumblebee/
|   |   ├── bumblebee.py
|   |   └── decoder_block.py
|   ├── optimus_prime/
|   |   ├── decoder_block.py
|   |   ├── encoder_block.py
|   |   └── optimus_prime.py
|   ├── layer_normalization.py
|   ├── loss.py
|   ├── mlp.py
|   ├── multiheaded_attention.py
|   └── positional_encoding.py
├── runs/
├── .gitignore
├── README.md
├── test.py
├── train.py
├── train_main.py
└── utils.py
```