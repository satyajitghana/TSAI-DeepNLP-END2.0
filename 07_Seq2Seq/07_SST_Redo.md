## 07 SST Model Redo (w/o Augmentation)

## Assignment

1.  Assignment 1 (500 points):
    1.  Submit the Assignment 5 as Assignment 1. To be clear,
        1.  ONLY use datasetSentences.txt. (no augmentation required)
        2.  Your dataset must have around 12k examples.
        3.  Split Dataset into 70/30 Train and Test (no validation)
        4.  Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        5.  Upload to github and proceed to answer these questions asked in the S7 - Assignment Solutions, where these questions are asked:
            1.  Share the link to your github repo (100 pts for code quality/file structure/model accuracy)
            2.  Share the link to your readme file (200 points for proper readme file)
            3.  Copy-paste the code related to your dataset preparation (100 pts)
            4.  Share your training log text (you MUST have been testing for test accuracy after every epoch) (200 pts)
            5.  Share the prediction on 10 samples picked from the test dataset. (100 pts)

## Solution

|| NBViewer | Google Colab | TensorBoard Logs |
|--|--|--| --|
|SST Dataset Preparation | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/SST_Redo/SST_Dataset_Augmentation.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/SST_Redo/SST_Dataset_Augmentation.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> |  |
|SST `Dataset` | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/SST_Redo/SST_Dataset.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/SST_Redo/SST_Dataset.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> |  |
|SST Model | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/SST_Redo/SSTModel.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/SST_Redo/SSTModel.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> | <a href="https://tensorboard.dev/experiment/h1GB1XeEQQKDGqTgXqVJgw/#scalars"><img src="https://img.shields.io/badge/logs-tensorboard-orange?logo=Tensorflow"></a> |


### Dataset Preparation

These are the following files that we will use in the Actual SST Dataset

1. `sentiment_labels.txt`: contains the `phrase_ids` and its respective `sentiment_values` which are in the range `[0,1]`
2. `datasetSentences.txt`: contains the sentences, and theirs ids, this in itself is not useful, because we dont have the label for the sentences
3. `dictionary.txt`: contains the `phrases` and its `phrase_ids`

So how do you get the label for the sentence?

you simply do `sentence` == `phrase` and map the sentences to phrase, so you know which sentence is which phrase, and then you will get the corresponding `phrase_id` and then you can use this to get the `label`

Read the labels file

```python
sentiment_labels = pd.read_csv(os.path.join(sst_dir,  "sentiment_labels.txt"), names=['phrase_ids',  'sentiment_values'], sep="|", header=0)
```

Discretize the labels to integers

```python
def discretize_label(label):
    if label <= 0.2: return 0
    if label <= 0.4: return 1
    if label <= 0.6: return 2
    if label <= 0.8: return 3
    return 4
```

```python
sentiment_labels['sentiment_values'] = sentiment_labels['sentiment_values'].apply(discretize_label)
```

Read the Sentences file

```python
sentence_ids = pd.read_csv(os.path.join(sst_dir,  "datasetSentences.txt"), sep="\t")
```

Read the phrase id to phrase mapping file
```python
dictionary = pd.read_csv(os.path.join(sst_dir,  "dictionary.txt"), sep="|", names=['phrase',  'phrase_ids'])
```

Read the train-test-dev split file
```python
train_test_split = pd.read_csv(os.path.join(sst_dir,  "datasetSplit.txt"))
```

This is important, there is where we are merging the dataframe !
```python
sentence_phrase_merge = pd.merge(sentence_ids, dictionary, left_on='sentence', right_on='phrase')
sentence_phrase_split = pd.merge(sentence_phrase_merge, train_test_split, on='sentence_index')
dataset = pd.merge(sentence_phrase_split, sentiment_labels, on='phrase_ids')
```

This is some cleaning to remove non-useful characters from the data
```python
dataset['phrase_cleaned'] = dataset['sentence'].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b",  lambda m: m.group(1))
```

This is how the dataframe looks like:

![enter image description here](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/assets/sst_df.png?raw=true)

And then we save this `DataFrame` and later simply read this file in our `Dataset` Class

Here is the PyTorch style `Dataset`

```python
class StanfordSentimentTreeBank(Dataset):
    """The Standford Sentiment Tree Bank Dataset
    Stanford Sentiment Treebank V1.0
    """

    ORIG_URL = "http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
    DATASET_NAME = "StanfordSentimentTreeBank"
    URL = 'https://drive.google.com/uc?id=1urNi0Rtp9XkvkxxeKytjl1WoYNYUEoPI'
    OUTPUT = 'sst_dataset.zip'
 

    def __init__(self, root, vocab=None, text_transforms=None, label_transforms=None, split='train', ngrams=1, use_transformed_dataset=True):
        """Initiate text-classification dataset.
        Args:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(self.__class__, self).__init__()

        if split not in ['train', 'test']:
            raise ValueError(f'split must be either ["train", "test"] unknown split {split}')

        self.vocab = vocab

        gdown.cached_download(self.URL, Path(root) / self.OUTPUT)

        self.generate_sst_dataset(split, Path(root) / self.OUTPUT)

        tokenizer = get_tokenizer("basic_english")

        # the text transform can only work at the sentence level
        # the rest of tokenization and vocab is done by this class
        self.text_transform = sequential_transforms(tokenizer, text_f.ngrams_func(ngrams))

        def build_vocab(data, transforms):
            def apply_transforms(data):
                for line in data:
                    yield transforms(line)
            return build_vocab_from_iterator(apply_transforms(data), len(data))

        if self.vocab is None:
            # vocab is always built on the train dataset
            self.vocab = build_vocab(self.dataset_train["phrase"], self.text_transform)


        if text_transforms is not None:
            self.text_transform = sequential_transforms(
                self.text_transform, text_transforms, text_f.vocab_func(self.vocab), text_f.totensor(dtype=torch.long)
            )
        else:
            self.text_transform = sequential_transforms(
                self.text_transform, text_f.vocab_func(self.vocab), text_f.totensor(dtype=torch.long)
            )

        self.label_transform = sequential_transforms(text_f.totensor(dtype=torch.long))

    def generate_sst_dataset(self, split, dataset_file):

        with ZipFile(dataset_file) as datasetzip:
            with datasetzip.open('sst_dataset/sst_dataset_augmented.csv') as f:
                dataset = pd.read_csv(f, index_col=0)

        self.dataset_orig = dataset.copy()

        dataset_train_raw = dataset[dataset['splitset_label'].isin([1, 3])]
        self.dataset_train = pd.concat([
                dataset_train_raw[['phrase_cleaned', 'sentiment_values']].rename(columns={"phrase_cleaned": 'phrase'}),
                dataset_train_raw[['synonym_sentences', 'sentiment_values']].rename(columns={"synonym_sentences": 'phrase'}),
                dataset_train_raw[['backtranslated', 'sentiment_values']].rename(columns={"backtranslated": 'phrase'}),
        ], ignore_index=True)

        if split == 'train':
            self.dataset = self.dataset_train.copy()
        else:
            self.dataset = dataset[dataset['splitset_label'].isin([2])] \
                                    [['phrase_cleaned', 'sentiment_values']] \
                                    .rename(columns={"phrase_cleaned": 'phrase'}) \
                                    .reset_index(drop=True)

    @staticmethod
    def discretize_label(label):
        if label <= 0.2: return 0
        if label <= 0.4: return 1
        if label <= 0.6: return 2
        if label <= 0.8: return 3
        return 4

    def __getitem__(self, idx):
        # print(f'text: {self.dataset["sentence"].iloc[idx]}, label: {self.dataset["sentiment_values"].iloc[idx]}')
        text = self.text_transform(self.dataset['phrase'].iloc[idx])
        label = self.label_transform(self.dataset['sentiment_values'].iloc[idx])
        # print(f't_text: {text} {text.shape}, t_label: {label}')
        return label, text 

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_labels():
        return ['very negative', 'negative', 'neutral', 'positive', 'very positive']

    def get_vocab(self):
        return self.vocab

    @property
    def collator_fn(self):
        def collate_fn(batch):
            pad_idx = self.get_vocab()['<pad>']
            labels, sequences = zip(*batch)
            labels = torch.stack(labels)
            lengths = torch.LongTensor([len(sequence) for sequence in sequences])
            sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                        padding_value = pad_idx,
                                                        batch_first=True 
            return labels, sequences, lengths
        return collate_fn
```

## Training Log

Tensorboard: https://tensorboard.dev/experiment/h1GB1XeEQQKDGqTgXqVJgw/#scalars

```
Best Test Acc: 29.99%
Params: 2.3M
Augmentations: NONE
Pretrained Embedding: NONE
```

Log

```
Epoch: 0, Test Acc: 0.2280784547328949, Test Loss: 1.4922568798065186
Validating: 100%
17/17 [00:00<00:00, 10.95it/s]
Epoch: 1, Test Acc: 0.2326740324497223, Test Loss: 1.4437384605407715
Validating: 100%
17/17 [00:00<00:00, 11.74it/s]
Epoch: 2, Test Acc: 0.23666085302829742, Test Loss: 1.422459363937378
Validating: 100%
17/17 [00:00<00:00, 11.58it/s]
Epoch: 3, Test Acc: 0.24293950200080872, Test Loss: 1.404987096786499
Validating: 100%
17/17 [00:00<00:00, 12.02it/s]
Epoch: 4, Test Acc: 0.249832883477211, Test Loss: 1.395544171333313
Validating: 100%
17/17 [00:00<00:00, 11.71it/s]
Epoch: 5, Test Acc: 0.2535093426704407, Test Loss: 1.3994262218475342
Validating: 100%
17/17 [00:00<00:00, 12.09it/s]
Epoch: 6, Test Acc: 0.2584153115749359, Test Loss: 1.4065455198287964
Validating: 100%
17/17 [00:00<00:00, 10.73it/s]
Epoch: 7, Test Acc: 0.2509012222290039, Test Loss: 1.3632327318191528
Validating: 100%
17/17 [00:00<00:00, 11.47it/s]
Epoch: 8, Test Acc: 0.2657622694969177, Test Loss: 1.382830262184143
Validating: 100%
17/17 [00:00<00:00, 10.22it/s]
Epoch: 9, Test Acc: 0.2766365110874176, Test Loss: 1.3973437547683716
Validating: 100%
17/17 [00:00<00:00, 10.86it/s]
Epoch: 10, Test Acc: 0.2795490324497223, Test Loss: 1.5025354623794556
Validating: 100%
17/17 [00:00<00:00, 10.16it/s]
Epoch: 11, Test Acc: 0.29301947355270386, Test Loss: 1.3790276050567627
Validating: 100%
17/17 [00:00<00:00, 11.44it/s]
Epoch: 12, Test Acc: 0.2876599431037903, Test Loss: 1.445764422416687
Validating: 100%
17/17 [00:00<00:00, 10.36it/s]
Epoch: 13, Test Acc: 0.2718857526779175, Test Loss: 1.649567723274231
Validating: 100%
17/17 [00:00<00:00, 11.76it/s]
Epoch: 14, Test Acc: 0.2856665253639221, Test Loss: 1.7272247076034546
Validating: 100%
17/17 [00:00<00:00, 10.58it/s]
Epoch: 15, Test Acc: 0.29959654808044434, Test Loss: 1.6855326890945435
Validating: 100%
17/17 [00:00<00:00, 10.71it/s]
Epoch: 16, Test Acc: 0.2838282883167267, Test Loss: 1.9708813428878784
Validating: 100%
17/17 [00:00<00:00, 11.56it/s]
Epoch: 17, Test Acc: 0.2920944094657898, Test Loss: 1.9789706468582153
Validating: 100%
17/17 [00:00<00:00, 10.09it/s]
Epoch: 18, Test Acc: 0.28826871514320374, Test Loss: 2.264657735824585
Validating: 100%
17/17 [00:00<00:00, 10.83it/s]
Epoch: 19, Test Acc: 0.28964143991470337, Test Loss: 2.3266751766204834
Validating: 100%
17/17 [00:00<00:00, 11.66it/s]
Epoch: 20, Test Acc: 0.29592007398605347, Test Loss: 2.2889909744262695
Validating: 100%
17/17 [00:00<00:00, 11.52it/s]
Epoch: 21, Test Acc: 0.2861260771751404, Test Loss: 2.5443098545074463
Validating: 100%
17/17 [00:00<00:00, 10.39it/s]
Epoch: 22, Test Acc: 0.2858157455921173, Test Loss: 2.618633985519409
Validating: 100%
17/17 [00:00<00:00, 10.77it/s]
Epoch: 23, Test Acc: 0.29700031876564026, Test Loss: 2.751962900161743
Validating: 100%
17/17 [00:00<00:00, 10.56it/s]
Epoch: 24, Test Acc: 0.29087090492248535, Test Loss: 2.886303663253784
Validating: 100%
17/17 [00:00<00:00, 10.03it/s]
Epoch: 25, Test Acc: 0.2911752760410309, Test Loss: 2.913256883621216
Validating: 100%
17/17 [00:00<00:00, 10.93it/s]
Epoch: 26, Test Acc: 0.28995177149772644, Test Loss: 3.2125508785247803
Validating: 100%
17/17 [00:00<00:00, 9.87it/s]
Epoch: 27, Test Acc: 0.28367310762405396, Test Loss: 3.3605377674102783
Validating: 100%
17/17 [00:00<00:00, 11.05it/s]
Epoch: 28, Test Acc: 0.2810649871826172, Test Loss: 3.2784640789031982
Validating: 100%
17/17 [00:00<00:00, 10.42it/s]
Epoch: 29, Test Acc: 0.28428784012794495, Test Loss: 3.468895435333252
```

## Sample Output

```
sentence:  Effective but too - tepid biopic
label: neutral, predicted: neutral


sentence:  The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
label: neutral, predicted: neutral


sentence:  Perhaps no picture ever made has more literally <unk> that the road to hell is paved with good intentions .
label: positive, predicted: positive


sentence:  Steers turns in a snappy screenplay that <unk> at the edges ; it 's so clever you want to hate it .
label: positive, predicted: positive


sentence:  This is a film well worth seeing , talking and singing heads and all .
label: very positive, predicted: very positive


sentence:  What really surprises about Wisegirls is its low - key quality and genuine tenderness .
label: positive, predicted: positive


sentence:  One of the greatest family - oriented , fantasy - adventure movies ever .
label: very positive, predicted: very positive


sentence:  Ultimately , it <unk> the reasons we need stories so much .
label: neutral, predicted: neutral


sentence:  An utterly compelling ` who wrote it ' in which the reputation of the most famous author who ever lived comes into question .
label: positive, predicted: positive


sentence:  A masterpiece four years in the making .
label: very positive, predicted: very positive

```
