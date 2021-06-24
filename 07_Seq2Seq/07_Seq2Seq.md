# 07 Seq2Seq

## Assignment

   2.  Assignment 2 (300 points):
        1.  Train **model we wrote in the class** on the following two datasets taken from  [this link (Links to an external site.)](https://kili-technology.com/blog/chatbot-training-datasets/):
            1.  [http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.)](http://www.cs.cmu.edu/~ark/QA-data/)
            2.  [https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.)](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
        2.  Once done, please upload the file to GitHub and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:
            1.  Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts)
            2.  Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)
            3.  Copy-paste the code related to your dataset preparation for both datasets. (100 pts)

[Update]

400 Points for each successful attempt on any additional dataset available on this link:  [https://kili-technology.com/blog/chatbot-training-datasetsLinks to an external site.](https://kili-technology.com/blog/chatbot-training-datasets)/

Share the notebook on which you attempted additional datasets successfully as your response to the assignment-solution page. Please make sure that you have explained the task and dataset that you have used.

## Solution

|| NBViewer | Google Colab | TensorBoard Logs |
|--|--|--| --|
|Wiki-QA Dataset | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/WikiQA_Dataset.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/WikiQA_Dataset.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> |
|Wiki-QA Model | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/WikiQA_Model.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/WikiQA_Model.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> | <a href="https://tensorboard.dev/experiment/ilMEHBPqQv6Tqh5DWC2SKw/"><img src="https://img.shields.io/badge/logs-tensorboard-orange?logo=Tensorflow"></a> |
|Quora-SQ Dataset | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/13876eec3594235c4df8545cf14bfd6328e01b8e/07_Seq2Seq/Quora_Question_Dataset.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/Quora_Question_Dataset.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> |
|Quora-SQ Model | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/QuoraSQ_Model.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/QuoraSQ_Model.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> | <a href="https://tensorboard.dev/experiment/9zZYZKtLQQmnOk4hpvgzsA/"><img src="https://img.shields.io/badge/logs-tensorboard-orange?logo=Tensorflow"></a> |



### Wiki QA Dataset


**Dataset Preparation**

```python
S08 = pd.read_csv('/content/Question_Answer_Dataset_v2.2/S10/question_answer_pairs.txt', sep='\t', encoding = "ISO-8859-1")
S09 = pd.read_csv('/content/Question_Answer_Dataset_v1.2/S09/question_answer_pairs.txt', sep='\t', encoding = "ISO-8859-1")
S10 = pd.read_csv('/content/Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt', sep='\t', encoding = "ISO-8859-1")
combined_qa = pd.concat([S08, S09, S10])
```

![wiki-qa](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/assets/wikiqa_df.png?raw=true)

The dataset is pretty straight forward, there are columns `Question` and `Answer` in each of the `S08`, `S09` and `S10` sub folder, you combine all of them. That's it. In total we have `3998` Rows and after dropping `NA` values we are left with `3422` Rows.

Which results in

![wiki-qa-cleaned](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/assets/wikiqa_cleaned.png?raw=true)

Below is a the PyTorch `Dataset` implementation of the same.

```python
class WikiQADataset(Dataset):
    """
    Wiki QA Dataset
    """

    URL = 'https://drive.google.com/uc?id=1FFTtPmxu63Dljelg8YsRRn8Yz475MWyv'
    OUTPUT = 'wikiqa_dataset.csv'
 

    def __init__(self, root, split='train', vocab=None, vectors=None, text_transforms=None, label_transforms=None, ngrams=1):
        """Initiate dataset.
        Args:
            vocab: Vocabulary object used for dataset.
        """

        super(self.__class__, self).__init__()

        if vectors:
            raise NotImplementedError(f'vectors not supported for this dataset as of now') 

        if split not in ['train', 'test']:
            raise ValueError(f'split must be either "train" or "test" unknown split {split}')

        if vocab and vectors:
            raise ValueError(f'both vocab and vectors cannot be provided')

        self.vocab = vocab
        self.vectors = vectors

        gdown.cached_download(self.URL, Path(root) / self.OUTPUT)

        self.generate_tweet_dataset(Path(root) / self.OUTPUT)

        self.train_dset, self.test_dset = train_test_split(self.full_dataset_, test_size=0.3, random_state=42)

        if split == 'train':
            self.dataset = self.train_dset
        elif split == 'test':
            self.dataset = self.test_dset
        else:
            raise ValueError("What did you do? you stupid potato?")

        # create the tokenizer, here we use spacy
        tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.tokenizer = tokenizer

        # the text transform can only work at the sentence level
        # the rest of tokenization and vocab is done by this class
        self.text_transform = text_f.sequential_transforms(tokenizer, text_f.ngrams_func(ngrams))

        self.vocab_transforms = text_f.sequential_transforms()
        self.vector_transforms = text_f.sequential_transforms()

        def build_vocab(data, transforms):
            def apply_transforms(data):
                for line in data:
                    yield transforms(line)
            return build_vocab_from_iterator(apply_transforms(data), len(data), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

        if self.vectors:
            self.vector_transforms = text_f.sequential_transforms(
                partial(vectors.get_vecs_by_tokens, lower_case_backup=True)
            )
        elif self.vocab is None:
            self.vocab = build_vocab(
                pd.concat([self.train_dset['Question'], self.train_dset['Answer']]),
                self.text_transform
            )
            self.PAD_IDX = self.vocab['<pad>']
            self.BOS_IDX = self.vocab['<bos>']
            self.EOS_IDX = self.vocab['<eos>']
    
        # if the user is using vocab, instead of vector
        if self.vocab:
            self.vocab_transforms = text_f.sequential_transforms(
                text_f.vocab_func(self.vocab), text_f.totensor(dtype=torch.long)
            )

        # label transform is similar to text_transform for this dataset except this does not have vectors
        self.label_transform = text_f.sequential_transforms(
            self.text_transform, self.vocab_transforms
        )

        if text_transforms is not None:
            self.text_transform = text_f.sequential_transforms(
                self.text_transform, text_transforms, self.vocab_transforms, self.vector_transforms 
            )
        else:
            self.text_transform = text_f.sequential_transforms(
                self.text_transform, self.vocab_transforms, self.vector_transforms
            )
        

    def generate_tweet_dataset(self, dataset_file):
        self.full_dataset_ = pd.read_csv(dataset_file)

    def __getitem__(self, idx):
        text = self.text_transform(self.dataset['Question'].iloc[idx])
        label = self.label_transform(self.dataset['Answer'].iloc[idx])
        return label, text

    def __len__(self):
        return len(self.dataset)

    def get_vocab(self):
        return self.vocab

    def get_vectors(self):
        return self.vectors

    def batch_sampler_fn(self):
        def batch_sampler():
            indices = [(i, len(self.tokenizer(s[1]))) for i, s in enumerate(train_list)]
            random.shuffle(indices)
            pooled_indices = []
            # create pool of indices with similar lengths 
            for i in range(0, len(indices), batch_size * 100):
                pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

            pooled_indices = [x[0] for x in pooled_indices]

            # yield indices for current batch
            for i in range(0, len(pooled_indices), batch_size):
                yield pooled_indices[i:i + batch_size]
        return batch_sampler()


    def collator_fn(self):
        def collate_fn(batch):
            targets, sequences = zip(*batch)
            lengths = torch.LongTensor([len(sequence) for sequence in sequences])
            targets = [torch.cat([torch.tensor([self.BOS_IDX]), item, torch.tensor([self.EOS_IDX])]) for item in targets]
            sequences = [torch.cat([torch.tensor([self.BOS_IDX]), item, torch.tensor([self.EOS_IDX])]) for item in sequences]

            if not self.vectors:
                pad_idx = self.PAD_IDX
                sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                            padding_value = pad_idx,
                                                            batch_first=True
                                                            )
                targets = torch.nn.utils.rnn.pad_sequence(targets, 
                                            padding_value = pad_idx,
                                            batch_first=True
                                            )
            
            return targets, sequences, lengths
        
        return collate_fn
```

### WikiQA Model - Training Log

```
  | Name    | Type             | Params
---------------------------------------------
0 | encoder | Encoder          | 5.1 M 
1 | decoder | Decoder          | 7.8 M 
2 | loss    | CrossEntropyLoss | 0     
---------------------------------------------
12.9 M    Trainable params
0         Non-trainable params
12.9 M    Total params
51.496    Total estimated model params size (MB)


Validating: 100%
9/9 [00:01<00:00, 5.60it/s]
Epoch: 0, Test PPL: 329.2035827636719, Test Loss: 5.630014896392822
Validating: 100%
9/9 [00:01<00:00, 5.33it/s]
Epoch: 1, Test PPL: 323.3686828613281, Test Loss: 5.590897560119629
Validating: 100%
9/9 [00:01<00:00, 5.32it/s]
Epoch: 2, Test PPL: 320.0765380859375, Test Loss: 5.55886173248291
Validating: 100%
9/9 [00:01<00:00, 5.89it/s]
Epoch: 3, Test PPL: 308.7172546386719, Test Loss: 5.521171569824219
Validating: 100%
9/9 [00:01<00:00, 6.69it/s]
Epoch: 4, Test PPL: 328.9159851074219, Test Loss: 5.575353622436523
Validating: 100%
9/9 [00:01<00:00, 5.37it/s]
Epoch: 5, Test PPL: 323.0148620605469, Test Loss: 5.544531345367432
Validating: 100%
9/9 [00:01<00:00, 5.33it/s]
Epoch: 6, Test PPL: 321.3564147949219, Test Loss: 5.539650917053223
Validating: 100%
9/9 [00:01<00:00, 5.77it/s]
Epoch: 7, Test PPL: 338.40802001953125, Test Loss: 5.586311340332031
Validating: 100%
9/9 [00:01<00:00, 6.56it/s]
Epoch: 8, Test PPL: 344.0347595214844, Test Loss: 5.59952449798584
Validating: 100%
9/9 [00:01<00:00, 5.30it/s]
Epoch: 9, Test PPL: 352.4696960449219, Test Loss: 5.6175217628479
```

### Quora SQ Dataset

This dataset is a `.tsv` file

```
id	qid1	qid2	question1	question2	is_duplicate
0	1	2	What is the step by step guide to invest in share market in india?	What is the step by step guide to invest in share market?	0
1	3	4	What is the story of Kohinoor (Koh-i-Noor) Diamond?	What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?	0
2	5	6	How can I increase the speed of my internet connection while using a VPN?	How can Internet speed be increased by hacking through DNS?	0
3	7	8	Why am I mentally very lonely? How can I solve it?	Find the remainder when [math]23^{24}[/math] is divided by 24,23?	0
4	9	10	Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?	Which fish would survive in salt water?	0
5	11	12	Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?	I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?	1
6	13	14	Should I buy tiago?	What keeps childern active and far from phone and video games?	0
7	15	16	How can I be a good geologist?	What should I do to be a great geologist?	1
8	17	18	When do you use シ instead of し?	"When do you use ""&"" instead of ""and""?"	0
```

The only columns we are interested is in are `question1`, `question2` and `is_duplicate`

Also we want only the questions that are duplicate, because our model will take in `question1` and try to generate `question`

```python
duplicate_df = quora_df[quora_df['is_duplicate'] == 1]
``` 

![quora_sq](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/07_Seq2Seq/assets/quorasq_df.png?raw=true)

This dataset was huge `＼（〇_ｏ）／` with over `149263 rows`, also this was a pain to train, compared to the other model.

The PyTorch style `Dataset` of this is exactly same as before dataset, just the `.csv` file is different. It would have been better to create a new Class `CSVDataset` for these kind of datasets.

TorchText does have a `TabularDataset` implementation, but that is old, and i refuse to use it. I will find a way to create a normal pytorch `Dataset` for Tabular Data.

```python
class QuoraSQDataset(Dataset):
    """
    Quora Similar Questions Dataset
    """

    URL = 'https://drive.google.com/uc?id=1g2YqSPXBWdCU1SjkCb69ENUuoEuxPbLg'
    OUTPUT = 'quora_duplicate_only_questions.csv'
 

    def __init__(self, root, split='train', vocab=None, vectors=None, text_transforms=None, label_transforms=None, ngrams=1):
        """Initiate dataset.
        Args:
            vocab: Vocabulary object used for dataset.
        """

        super(self.__class__, self).__init__()

        if vectors:
            raise NotImplementedError(f'vectors not supported for this dataset as of now') 

        if split not in ['train', 'test']:
            raise ValueError(f'split must be either "train" or "test" unknown split {split}')

        if vocab and vectors:
            raise ValueError(f'both vocab and vectors cannot be provided')

        self.vocab = vocab
        self.vectors = vectors

        gdown.cached_download(self.URL, Path(root) / self.OUTPUT)

        self.generate_tweet_dataset(Path(root) / self.OUTPUT)

        self.train_dset, self.test_dset = train_test_split(self.full_dataset_, test_size=0.3, random_state=42)

        if split == 'train':
            self.dataset = self.train_dset
        elif split == 'test':
            self.dataset = self.test_dset
        else:
            raise ValueError("What did you do? you stupid potato?")

        # create the tokenizer, here we use spacy
        tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.tokenizer = tokenizer

        # the text transform can only work at the sentence level
        # the rest of tokenization and vocab is done by this class
        self.text_transform = text_f.sequential_transforms(tokenizer, text_f.ngrams_func(ngrams))

        self.vocab_transforms = text_f.sequential_transforms()
        self.vector_transforms = text_f.sequential_transforms()

        def build_vocab(data, transforms):
            def apply_transforms(data):
                for line in data:
                    yield transforms(line)
            return build_vocab_from_iterator(apply_transforms(data), len(data), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

        if self.vectors:
            self.vector_transforms = text_f.sequential_transforms(
                partial(vectors.get_vecs_by_tokens, lower_case_backup=True)
            )
        elif self.vocab is None:
            self.vocab = build_vocab(
                pd.concat([self.train_dset['question1'], self.train_dset['question2']]),
                self.text_transform
            )
            self.PAD_IDX = self.vocab['<pad>']
            self.BOS_IDX = self.vocab['<bos>']
            self.EOS_IDX = self.vocab['<eos>']

        # if the user is using vocab, instead of vector
        if self.vocab:
            self.vocab_transforms = text_f.sequential_transforms(
                text_f.vocab_func(self.vocab), text_f.totensor(dtype=torch.long)
            )

        # label transform is similar to text_transform for this dataset except this does not have vectors
        self.label_transform = text_f.sequential_transforms(
            self.text_transform, self.vocab_transforms
        )

        if text_transforms is not None:
            self.text_transform = text_f.sequential_transforms(
                self.text_transform, text_transforms, self.vocab_transforms, self.vector_transforms 
            )
        else:
            self.text_transform = text_f.sequential_transforms(
                self.text_transform, self.vocab_transforms, self.vector_transforms
            )
        

    def generate_tweet_dataset(self, dataset_file):
        self.full_dataset_ = pd.read_csv(dataset_file)

    def __getitem__(self, idx):
        text = self.text_transform(self.dataset['question1'].iloc[idx])
        label = self.label_transform(self.dataset['question2'].iloc[idx])
        return label, text

    def __len__(self):
        return len(self.dataset)

    def get_vocab(self):
        return self.vocab

    def get_vectors(self):
        return self.vectors

    def batch_sampler_fn(self):
        def batch_sampler():
            indices = [(i, len(self.tokenizer(s[1]))) for i, s in enumerate(train_list)]
            random.shuffle(indices)
            pooled_indices = []
            # create pool of indices with similar lengths 
            for i in range(0, len(indices), batch_size * 100):
                pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

            pooled_indices = [x[0] for x in pooled_indices]

            # yield indices for current batch
            for i in range(0, len(pooled_indices), batch_size):
                yield pooled_indices[i:i + batch_size]
        return batch_sampler()


    def collator_fn(self):
        def collate_fn(batch):
            targets, sequences = zip(*batch)
            lengths = torch.LongTensor([len(sequence) for sequence in sequences])
            targets = [torch.cat([torch.tensor([self.BOS_IDX]), item, torch.tensor([self.EOS_IDX])]) for item in targets]
            sequences = [torch.cat([torch.tensor([self.BOS_IDX]), item, torch.tensor([self.EOS_IDX])]) for item in sequences]

            if not self.vectors:
                pad_idx = self.PAD_IDX
                sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                            padding_value = pad_idx,
                                                            batch_first=True
                                                            )
                targets = torch.nn.utils.rnn.pad_sequence(targets, 
                                            padding_value = pad_idx,
                                            batch_first=True
                                            )
            
            return targets, sequences, lengths
        
        return collate_fn
```

### QuoraSQ Model - Training Log

```
  | Name    | Type             | Params
---------------------------------------------
0 | encoder | Encoder          | 2.2 M 
1 | decoder | Decoder          | 4.3 M 
2 | loss    | CrossEntropyLoss | 0     
---------------------------------------------
6.5 M     Trainable params
0         Non-trainable params
6.5 M     Total params
26.076    Total estimated model params size (MB)

Epoch: 0, Test PPL: 251.0346221923828, Test Loss: 5.5212016105651855
Epoch: 1, Test PPL: 214.76112365722656, Test Loss: 5.36377477645874
Epoch: 2, Test PPL: 187.0980987548828, Test Loss: 5.2250542640686035
Epoch: 3, Test PPL: 152.59373474121094, Test Loss: 5.019686222076416
Epoch: 4, Test PPL: 133.257568359375, Test Loss: 4.8832526206970215
```



## Takeaways

- There was no code-directory structure followed. I was planning to do it, but my lazy a didn't do it. The idea is to have a python package to store all the models and the datasets and then you can simply import them and train them. I've done it before for Vision based Models/Dataset, so this should also be simple, just that it's too time taking and i feel lazy doing the same thing over `≧ ﹏ ≦`
- I need to fix the `batch_sampler` so it works just like the `BucketIterator`


---

<center>
<iframe src="https://giphy.com/embed/bOwOAey4MDO3ivBkgK" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/end-old-hollywood-fin-bOwOAey4MDO3ivBkgK"></a></p>
</center>

