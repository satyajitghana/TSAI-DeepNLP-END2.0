# 08 TorchText

## Assignment

Refer to this  [Repo (Links to an external site.)](https://github.com/bentrevett/pytorch-seq2seq).

1.  You are going to refactor this repo in the next 3 sessions. In the current assignment, change the 2 and 3 (optional 4, 500 additional points) such that
    1.  is uses none of the legacy stuff
    2.  It MUST use Multi30k dataset from torchtext
    3.  uses yield_token, and other code that we wrote
2.  Once done, proceed to answer questions in the Assignment-Submission Page.

## Solution


|| NBViewer | Google Colab |
|--|--|--|
|1 - Sequence to Sequence Learning with Neural Networks | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> 
|2 - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> 
|3 - Neural Machine Translation by Jointly Learning to Align and Translate | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/3_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/3_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> 
|4 - Packed Padded Sequences, Masking, Inference and BLEU | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/4_Packed_Padded_Sequences%2C_Masking%2C_Inference_and_BLEU.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/08_TorchText/pytorch-seq2seq-modern/4_Packed_Padded_Sequences%2C_Masking%2C_Inference_and_BLEU.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> 

## The New API

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
```

Note that there are more api's coming up in `torchtext.experimental` which will be released soon in `0.11.0` and `0.12.0`, hopefully with good documentation :) and i'll be a part of creating that documentation 😁

1. This is how tokenizers are instantiated now

```python
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
```

2. Loading the dataset

```python
# Training, Validation and Test data Iterator
train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_list, val_list, test_list = list(train_iter), list(val_iter), list(test_iter)
```

A thing to note here is that `list(iter)` will be costly for large dataset, so its preferable to keep it as an `iter` and make a `Dataloader` out of it and use the `Dataloader` for whatever you want to do, because once the iterator is exhausted, you'll need to call the `Multi30k` function again to make the iter, which seems kind of waste of cpu cycles.

3. Creating the `Vocab`

```python
# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_list, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)
```

4. Creating the transforms and `collate_fn`

```python
from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, src_len, tgt_batch = [], [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        src_len.append(len(src_batch[-1]))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, torch.LongTensor(src_len), tgt_batch
```

5. `Dataloader`

```python
BATCH_SIZE = 128

train_dataloader = DataLoader(train_list, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_dataloader = DataLoader(val_list, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_list, batch_size=BATCH_SIZE, collate_fn=collate_fn)
```

The main aim of the new API is to be consistent with `torch`, which uses a classical `Dataloader` object, and `torchtext` is moving towards it. UNIFY EVERYTHING `(╯°□°）╯︵ ┻━┻`

## Notes

`torchtext` is evolving a lott, consequently there has been a lot of breaking changes, and not much documentation on it sadly `(┬┬﹏┬┬)`, below are some of the official reference which use the new api

- [transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [seq2seq translation tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [translation transformer](https://pytorch.org/tutorials/beginner/translation_transformer.html)
- [`torchtext` legacy migration tutorial](https://github.com/pytorch/text/blob/release/0.9/examples/legacy_tutorial/migration_tutorial.ipynb)

Apart from these there are some useful GitHub Issues that must be looked at. Some of them are my contributions `ψ(｀∇´)ψ`

- [build vocab from GloVe Embeddings](https://github.com/pytorch/text/issues/1350)
- [update on package documentation](https://github.com/pytorch/text/issues/1349)
- [how to use `Vector` with the new `torchtext` api](https://github.com/pytorch/text/issues/1323)

---

<center>
What it feels like to transition from `torchtext.legacy` to `torchtext`
<iframe src="https://giphy.com/embed/1gdkNV0hroZjOXSHOU" width="480" height="350" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/boomerangtoons-mad-upset-1gdkNV0hroZjOXSHOU"></a></p>
</center>
