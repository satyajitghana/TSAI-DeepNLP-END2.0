
# 12 Attention Is All You Need

## Assignment

The code we are referring to comes from  [here (Links to an external site.)](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/a2ee9271b5280be6994660c7982d0f44c67c3b63/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py).

Take the code above, and make it work with any dataset. Submit the GitHub repo's ReadMe file, where I can see answers to these questions:

-   what dataset you have used
-   what problem have you solved (fill in the blank, translation, text generation, etc)
-   the output of your training for 10 epochs

## Solution

|| NBViewer | Google Colab | Tensorboard Logs
|--|--|--|--|
| Attention Is All You Need - **Solution** | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/12_AttentionIsAllYouNeed/Attention_Is_All_You_Need.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/12_AttentionIsAllYouNeed/Attention_Is_All_You_Need.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> | <a href="https://tensorboard.dev/experiment/pqsWXkyvS2umPiN8rr1snw/"><img src="https://img.shields.io/badge/logs-tensorboard-orange?logo=Tensorflow"></a>

## Dataset

Dataset used was **Multi30K**, but the DataModule also supports **IWSLT2016**

```python
class TTCTranslation(pl.LightningDataModule):

    DATASET_OPTIONS = ['multi30k', 'iwslt2016']
    LANGUAGE_OPTIONS = ['en', 'fr', 'de', 'cs', 'ar']

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, language_pair=('en', 'de'), spacy_language_pair=('en_core_web_sm', 'de_core_news_sm'), dataset='multi30k', batch_size=64, batch_first=True):
        super().__init__()

        assert len(language_pair) == 2 and len(spacy_language_pair), f'tf are you doing? give me a language \"pair\"'
        assert dataset in self.DATASET_OPTIONS, f'{self.DATASET_OPTIONS} are only supported'
        assert all(x in self.LANGUAGE_OPTIONS for x in language_pair), f'{self.LANGUAGE_OPTIONS} are only supported'

        self.batch_size = batch_size
        self.batch_first = batch_first

        self.language_pair = language_pair
        self.src_lang, self.tgt_lang = language_pair
        self.spacy_src_lang, self.spacy_tgt_lang = spacy_language_pair

        if dataset == 'multi30k':
            self.train_dataset = Multi30k(split='train', language_pair=self.language_pair)
            self.val_dataset = Multi30k(split='valid', language_pair=self.language_pair)
            self.test_dataset = Multi30k(split='test', language_pair=self.language_pair)
        elif dataset == 'iwslt2016':
            self.train_dataset = IWSLT2016(split='train', language_pair=self.language_pair)
            self.val_dataset = IWSLT2016(split='valid', language_pair=self.language_pair)
            self.test_dataset = IWSLT2016(split='test', language_pair=self.language_pair)
        
        self.train_dataset, self.val_dataset, self.test_dataset = list(self.train_dataset), list(self.val_dataset), list(self.test_dataset)

        # --- token transform ---

        self.token_transform = {}
        self.token_transform[self.src_lang] = get_tokenizer('spacy', language=self.spacy_src_lang)
        self.token_transform[self.tgt_lang] = get_tokenizer('spacy', language=self.spacy_tgt_lang)

        # --- vocab transform ---
        # helper function to yield list of tokens
        def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
            language_index = {self.src_lang: 0, self.tgt_lang: 1}

            for data_sample in data_iter:
                yield self.token_transform[language](data_sample[language_index[language]])
        
        self.vocab_transform = {}
        for ln in self.language_pair:
            # Create torchtext's Vocab object 
            self.vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(self.train_dataset, ln),
                                                            min_freq=1,
                                                            specials=self.SPECIAL_SYMBOLS,
                                                            special_first=True)

        # Set UNK_IDX as the default index. This index is returned when the token is not found. 
        # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary. 
        for ln in self.language_pair:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)

        # --- text/tensor transform ---

        # function to add BOS/EOS and create tensor for input sequence indices
        def tensor_transform(token_ids: List[int]):
            return torch.cat((torch.tensor([self.BOS_IDX]), 
                            torch.tensor(token_ids), 
                            torch.tensor([self.EOS_IDX])))

        # src and tgt language text transforms to convert raw strings into tensors indices
        self.text_transform = {}
        for ln in self.language_pair:
            self.text_transform[ln] = sequential_transforms(self.token_transform[ln], #Tokenization
                                                    self.vocab_transform[ln], #Numericalization
                                                    tensor_transform) # Add BOS/EOS and create tensor

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator_fn 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator_fn
        )

    @property
    def collator_fn(self):
        def wrapper(batch):
            src_batch, tgt_batch = [], []
            for src_sample, tgt_sample in batch:
                src_batch.append(self.text_transform[self.src_lang](src_sample.rstrip("\n")))
                tgt_batch.append(self.text_transform[self.tgt_lang](tgt_sample.rstrip("\n")))

            src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=self.PAD_IDX, batch_first=self.batch_first)
            tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.PAD_IDX, batch_first=self.batch_first)

            return src_batch, tgt_batch

        return wrapper

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass
```

## Problem Solved

Transformers can be used to solve various NLP Tasks like (fill in the blank, translation, text generation, etc), in this notebook i used to to solve Language Translation Problem.

```python
sentence = "Two young, White males are outside near many bushes."
translate_sentence(transformer, ttc_translation, sentence)

>>> 'Zwei junge Männer sind in weißen Nähe von Büschen .'
```

## Training Logs

Tensorboard Logs: https://tensorboard.dev/experiment/pqsWXkyvS2umPiN8rr1snw/




