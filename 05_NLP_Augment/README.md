# 05 NLP Augment

## Assignment

1.  Look at  [this code (Links to an external site.)](https://colab.research.google.com/drive/19wZi7P0Tzq9ZxeMz5EDmzfWFBLFWe6kN?usp=sharing&pli=1&authuser=3)  above. It has additional details on "Back Translate", i.e. using Google translate to convert the sentences. It has "random_swap" function, as well as "random_delete".
2.  Use "Back Translate", "random_swap" and "random_delete" to augment the data you are training on
3.  Download the StanfordSentimentAnalysis Dataset from this  [link (Links to an external site.)](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)(it might be troubling to download it, so force download on chrome). Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.
4.  Train your model and achieve  **60%+ validation/test accuracy**. Upload your collab file on GitHub with readme that contains details about your assignment/word (minimum  **250 words**),  **training logs showing final validation accuracy, and outcomes for  10  example inputs from the test/validation data.**
5.  **You must submit before DUE date (and not "until" date).**

# Solution

| Dataset | Augmentation | Model |
|--|--|--|
| [Github](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/SST_Dataset.ipynb) | [Github](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/SST_Dataset_Augmentation.ipynb) | [Github](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/SSTModel.ipynb)
| [Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/SST_Dataset.ipynb) | [Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/SST_Dataset_Augmentation.ipynb)  | [Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/SSTModel.ipynb)

Tensorboard Experiment: https://tensorboard.dev/experiment/8LTXbHV8QQGXxBASaVYeJw/#scalars

---
Note about Dataset

There was this gist https://gist.github.com/wpm/52758adbf506fd84cff3cdc7fc109aad
which claims to parse the SST dataset properly, but there are comments on the gist like
>This script make unusual thing - it pushes all non-sentence phrases from dictionary to train sample. So you will achive training sample with 230K trees inside. I've spent some time before notice this. Be careful

Which is why this way was NOT considered, Instead I matched the phrases to the sentences, and based on that the label was got. Individual phrases were not included in training set. ONLY the sentences and their labels were included. Ofcourse with a lot of augmentations `○( ＾皿＾)っ Hehehe…`

## NLP Augmentations

1. traduction inversée
(Back Translation)

This was probably very tricky, and difficult to do. I was using `nlpaug` a package to simplify nlp augmentations and i used google translate python library to translate sentences, which basically internally using HTTP/2 API calls to translate.

Here's the code for that

```python
from nlpaug.augmenter.word import WordAugmenter

import google_trans_new
from google_trans_new import google_translator  

import random

class BackTranslateAug(WordAugmenter):
    def __init__(self, name='BackTranslateAug', aug_min=1, aug_max=10, 
                 aug_p=0.3, stopwords=None, tokenizer=None, reverse_tokenizer=None, 
                 device='cpu', verbose=0, stopwords_regex=None):
        super(BackTranslateAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_min=aug_min, aug_max=aug_max, 
                 aug_p=aug_p, stopwords=stopwords, tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, 
                 device=device, verbose=0, stopwords_regex=stopwords_regex)
        

        self.translator = google_translator()
        
    def substitute(self, data):
        if not data:
            return data
            
        if self.prob() < self.aug_p:
            trans_lang = random.choice(list(google_trans_new.LANGUAGES.keys()))
            trans_text = self.translator.translate(data, lang_src='en', lang_tgt=trans_lang) 

            en_text = self.translator.translate(trans_text, lang_src=trans_lang, lang_tgt='en') 

            return en_text

        return data
```

```python
aug = BackTranslateAug(aug_max=3, aug_p=1)
augmented_text = aug.augment(text)
```

```
Original: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal . 
Augmented Text: The Rock is intended to be the 21st century new `` Conan 'and that he will do a splash even larger than Arnold Schwarzenegger, Jean-Claud Van Damme or Steven Segal.
```

Seems straight forward right ? just apply this over the entire `DataFrame` ?
**WRONG**

`8% 958/11286 [09:54<2:13:42, 1.29it/s]`

It would take about 2 hours to do this, even on an 8 core beast machine with really good internet. So i even tried different multiprocessing libraries, but they don't work, it the google translate api library gets locked up. And i tested this on colab and also a standalone pc. same issue.

ALSO you will exhaust the requests limit

```
[/usr/local/lib/python3.7/dist-packages/google_trans_new/google_trans_new.py](https://localhost:8080/#) in translate(self, text, lang_tgt, lang_src, pronounce)  192  except  requests.exceptions.HTTPError  as  e:  193  # Request successful, bad response  --> 194  raise  google_new_transError(tts=self,  response=r)  195  except  requests.exceptions.RequestException  as  e:  196  # Request failed  

google_new_transError: 429 (Too Many Requests) from TTS API. Probable cause: Unknown
```

So what can you do ?

**PLAY SMORT**

Beat Google at their own game.

When i was struggling with this, i realised something, i remembered that there's google translate api built into Google Sheets, so here i come, using Google Sheets as a NLP Data Augmentor.

![gsheet augmentor](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/gsheet_translate.png?raw=true)

It took about 60-70 mins, but it was done atleast `(～￣▽￣)～`

[Link to Google Sheet Back Translate Augmentor](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5G4wKHEXkseaSy_8khXdmUqfx2jVUK4T-ITSeq8AMB1QWJoyZrpzelCf8Sb70mhs0knjqCEdZguWT/pubhtml)

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5G4wKHEXkseaSy_8khXdmUqfx2jVUK4T-ITSeq8AMB1QWJoyZrpzelCf8Sb70mhs0knjqCEdZguWT/pubhtml?widget=true&amp;headers=false" width="700" height="500"></iframe>

2. Synonym Augment

Substitutes a random word with their synonym

```python
aug = naw.SynonymAug(aug_src='wordnet')
synonym_sentences = dataset_aug['sentence'].progress_apply(aug.augment)
```

```
Original: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal . 
Augmented Text: The Rock is destined to follow the 21st Hundred ' s new ` ` Conan ' ' and that helium ' s going to make a splash yet swell than Arnold Schwarzenegger, Blue jean - Claud Van Damme or Steven Segal.
```

3. Random Delete

```python
aug = naw.RandomWordAug(aug_max=3)
augmented_text = aug.augment(text)
```

```
Original: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal . 
Augmented Text: The Rock is destined to be the 21st ' s new ` ` Conan ' ' that ' s going to make a splash even greater than Arnold Schwarzenegger, Jean - Claud Van Damme or Steven Segal.
```

4. Random Swap

```python
aug = naw.RandomWordAug(action="swap", aug_max=3)
augmented_text = aug.augment(text)
```

```
Original: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal . 
Augmented Text: The Rock is destined to be the 21st Century ' s new ` ` Conan ' ' and he that ' s going to make a splash even greater than Arnold, Schwarzenegger Jean Claud - Van Damme or Steven Segal.
```

## El Modelo

Why did I stack 5 layers of LSTM ?

😅 I am new to NLP, i should have done research, so it turns out for classification tasks taking 2-3 layers is enough, we generally use more LSTM layers for sequence generation like machine translation.

>While it is not theoretically clear what is the additional power gained by the deeper architecture, it was observed empirically that deep RNNs work better than shallower ones on some tasks. In particular, Sutskever et al (2014) report that a 4-layers deep architecture was crucial in achieving good machine-translation performance in an encoder-decoder framework. Irsoy and Cardie (2014) also report improved results from moving from a one-layer BI-RNN to an architecture with several layers. Many other works report result using layered RNN architectures, but do not explicitly compare to 1-layer RNNs.

![model](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/05_NLP_Augment/model.png?raw=true)

## Experiments

| Model `[embedding_dim, dropout]`| LSTM `[hidden_dim,layers]`   | Augmentation | Epochs | Test Accuracy | Remark |
|--|--|--|--|--|--|
| 128, 0.2 | 256, 5 | delete, swap | 100 |40.8| Heavy Overfit
|
| 128, 0.2 | 256, 5 | delete, swap, synonym, translate | 30 |40.3| More Augmentation, Heavy Overfit
|
| 128, 0.2 | 256, 2 | delete, swap, synonym, translate | 30 |40.2| Less Layers, Heavy Overfit
|
| 128, 0.5 | 256, 2 | delete, swap, synonym, translate | 30 |39.7| Increased Dropout, Still Heavy Overfit
|
| 128, 0.5 | 128, 2 | delete, swap, synonym, translate | 30 |40.9| Decreased `hidden_dim`, Reduced Overfit
|
| 128, 0.5 | 64, 2 | delete, swap, synonym, translate | 30 |42.2| Decreased `hidden_dim`, Reduced Overfit
|
| 128, 0.5 | 32, 2 | delete, swap, synonym, translate | 30 |40.2| Decreased `hidden_dim`, Acc Reduced
|
| 128, 0.5 | 64, 5 | delete, swap, synonym, translate | 30 |40.5| Increased Num Layers
|
| 128, 0.0 | 64, 1 | delete, swap, synonym, translate | 30 |40.1| Single Layer LSTM, 0 Dropout
|

The logs can be viewed at https://tensorboard.dev/experiment/8LTXbHV8QQGXxBASaVYeJw/#scalars

Notice `version_5` above

- I forgot to take max of test accuracies `(。﹏。*)` so the test accuracy shown in the plot is the last epoch test accuracy, instead the validation accuracy can be considered.
- The epochs was reduced to 30, because now with augmentation it was 27K datapoints, so its almost 3 times as much, the epochs was reduced approx 3 times ~ 30 epochs.
- It takes only about **5 epochs** to reach **39% accuracy**, but from here on it gets difficult and the model starts overfitting.
- As soon as i saw reduced overfitting, i also saw that the model validation accuracy does not shake that much, it goes to 39 and stays there
- Adding more layers did not help at all, i should have used resnet style identity layers, could help.

## Misclassifications

```
sentence:  the film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
label: neutral, predicted: positive


sentence:  offers that rare combination of entertainment and education .
label: very positive, predicted: positive


sentence:  perhaps no picture ever made has more literally showed that the road to hell is paved with good intentions .
label: positive, predicted: neutral


sentence:  steers turns in a snappy screenplay that <unk> at the edges it ' s so clever you want to hate it .
label: positive, predicted: neutral


sentence:  but he somehow pulls it off .
label: positive, predicted: neutral


sentence:  take care of my cat offers a refreshingly different slice of asian cinema .
label: positive, predicted: very positive


sentence:  ultimately , it <unk> the reasons we need stories so much .
label: neutral, predicted: negative


sentence:  the movie ' s ripe , <unk> beauty will tempt those willing to probe its inscrutable mysteries .
label: positive, predicted: very positive


sentence:  offers a breath of the fresh air of true sophistication .
label: very positive, predicted: positive


sentence:  a disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by philip glass .
label: neutral, predicted: very positive
```

## Correct Classifications

```
sentence:  effective but <unk> biopic
label: neutral, predicted: neutral


sentence:  if you sometimes like to go to the movies to have fun , wasabi is a good place to start .
label: positive, predicted: positive


sentence:  emerges as something rare , an issue movie that ' s so honest and keenly observed that it doesn ' t feel like one .
label: very positive, predicted: very positive


sentence:  this is a film well worth seeing , talking and singing heads and all .
label: very positive, predicted: very positive


sentence:  what really surprises about wisegirls is its low-key quality and genuine tenderness .
label: positive, predicted: positive


sentence:  <unk> wendigo is <unk> why we go to the cinema to be fed through the eye , the heart , the mind .
label: positive, predicted: positive


sentence:  one of the greatest family-oriented , fantasy-adventure movies ever .
label: very positive, predicted: very positive


sentence:  an utterly compelling ` who wrote it ' in which the reputation of the most famous author who ever lived comes into question .
label: positive, predicted: positive


sentence:  illuminating if overly talky documentary .
label: neutral, predicted: neutral


sentence:  a masterpiece four years in the making .
label: very positive, predicted: very positive
```

## Learnings

- Learn how to use CometML or W&B, so i can save even the notebook code along with the hparam tuning values
