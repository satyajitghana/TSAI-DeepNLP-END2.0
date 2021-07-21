
# 10 Seq2Seq Attention

## Assignment

1.  Replace the embeddings of this session's code with GloVe embeddings
2.  Compare your results with this session's code.
3.  Upload to a public GitHub repo and proceed to Session 10 Assignment Solutions where these questions are asked:
    1.  Share the link to your README file's public repo for this assignment. Expecting a minimum 500-word write-up on your learnings. Expecting you to compare your results with the code covered in the class. - 750 Points
    2.  Share the link to your main notebook with training logs - 250 Points

## Solution

|| NBViewer | Google Colab |
|--|--|--|
|Old Code - French to English | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/END2_Translation_using_Seq2Seq_and_Attention.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/END2_Translation_using_Seq2Seq_and_Attention.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
|**New Code** - English to French w/ GloVe Embeddings | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/Seq2Seq_Attention.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/Seq2Seq_Attention.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

If someday PyTorch decides to remove the `data.zip` file, I've added it to [this repository](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/data.zip).

### Creating the Dataset

Some of the dataset code was changed so that it supports the PyTorch Lightning Data Module and Model, like

Use `build_vocab_from_iterator` to build a `Vocab` object, this will later be used with pretrained word embedding, to map the vocab to the GloVe's vocab.

```python
    def prepare_langs(self, lang_file='eng-fra', reverse=True):
        with urlopen(self.zip_url) as f:
            with BytesIO(f.read()) as b, ZipFile(b) as datazip:
                lang1, lang2 = lang_file.split('-')
                pairs = readPairs(datazip, lang1, lang2, reverse)

        print("Read %s sentence pairs" % len(pairs))
        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        input_sentences, target_sentences = zip(*pairs)

        input_lang = build_vocab_from_iterator(
            [sentence.split(' ') for sentence in input_sentences],
            specials=special_tokens
        )

        output_lang = build_vocab_from_iterator(
            [sentence.split(' ') for sentence in target_sentences],
            specials=special_tokens
        )

        setattr(input_lang, 'name', lang2 if reverse else lang1)
        setattr(output_lang, 'name', lang1 if reverse else lang2)

        setattr(input_lang, 'n_words', len(input_lang))
        setattr(output_lang, 'n_words', len(output_lang))


        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        return input_lang, output_lang, pairs

```

### Encoder and Decoder

The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

If only the context vector is passed between the encoder and decoder, that single vector carries the burden of encoding the entire sentence.

Attention allows the decoder network to "focus" on a different part of the encoder's outputs for every step of the decoder's own outputs. First we calculate a set of  _attention weights_. These will be multiplied by the encoder output vectors to create a weighted combination. The result (called  `attn_applied`  in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.

Calculating the attention weights is done with another feed-forward layer `attn`, using the decoder's input and hidden state as inputs. Because there are sentences of all sizes in the training data, to actually create and train this layer we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.

### Using Pretrained `GloVe` Embeddings

**Glo**bal **Ve**ctors for Word Representation, or GloVe, is an “[unsupervised learning algorithm for obtaining vector representations for words.](https://nlp.stanford.edu/projects/glove/)” Simply put, GloVe allows us to take a corpus of text, and intuitively transform each word in that corpus into a position in a high-dimensional space. This means that similar words will be placed together.

I found this nice way for using `Embeddings` with `GloVe` `Vectors`

```python
from torchtext.vocab import GloVe, vocab
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn as nn

#define your model that accepts pretrained embeddings 
class TextClassificationModel(nn.Module):

    def __init__(self, pretrained_embeddings, num_class, freeze_embeddings = False):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings, freeze = freeze_embeddings, sparse=True)
        self.fc = nn.Linear(pretrained_embeddings.shape[1], num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

train_iter = AG_NEWS(split = 'train')
num_class = len(set([label for (label, _) in train_iter]))
unk_token = "<unk>"
unk_index = 0
glove_vectors = GloVe()
glove_vocab = vocab(glove_vectors.stoi)
glove_vocab.insert_token("<unk>",unk_index)
#this is necessary otherwise it will throw runtime error if OOV token is queried 
glove_vocab.set_default_index(unk_index)
pretrained_embeddings = glove_vectors.vectors
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))

#instantiate model with pre-trained glove vectors
glove_model = TextClassificationModel(pretrained_embeddings, num_class)

tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split = 'train')
example_text = next(train_iter)[1]
tokens = tokenizer(example_text)
indices = glove_vocab(tokens)
text_input = torch.tensor(indices)
offset_input = torch.tensor([0])

model_output = glove_model(text_input, offset_input)
```

[Source](https://github.com/pytorch/text/issues/1350)

And for using Pretrained Embeddings with an existing Vocab object

```python
min_freq = 5
special_tokens = ['<unk>', '<pad>']

vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                                  min_freq=min_freq,
                                                  specials=special_tokens)

# train_data['tokens'] is a list of a list of strings, i.e. [['hello', 'world'], ['goodbye', 'moon']], where ['hello', 'moon'] is the tokens corresponding to the first example in the training set.

pretrained_vectors = torchtext.vocab.FastText()

pretrained_embedding = pretrained_vectors.get_vecs_by_tokens(vocab.get_itos())

# vocab.get_itos() returns a list of strings (tokens), where the token at the i'th position is what you get from doing vocab[token]
# get_vecs_by_tokens gets the pre-trained vector for each string when given a list of strings
# therefore pretrained_embedding is a fully "aligned" embedding matrix

class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.embedding(text)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction

vocab_size = len(vocab)
embedding_dim = 300
output_dim = n_classes

model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)

# super basic model here, important thing is the nn.Embedding layer that needs to have an embedding layer that is initialized as nn.Embedding(vocab_size, embedding_dim) with embedding_dim = 300 as that's the dimensions of the FastText embedding

model.embedding.weight.data = pretrained_embedding

# overwrite the model's initial embedding matrix weights with that of the pre-trained embeddings from FastText
```

And this is how I integrated GloVe Embeddings into this assignment

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_pretrained=False, vocab_itos=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if use_pretrained and vocab_itos is None:
            raise ValueError('`use_pretained=True` with `vocab_itos=None`, please provide the vocab itos List')
        
        if use_pretrained:
            glove_vec = torchtext.vocab.GloVe(name='6B')
            glove_emb = glove_vec.get_vecs_by_tokens(vocab_itos)
            self.embedding = nn.Embedding.from_pretrained(glove_emb, padding_idx=PAD_token)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

        assert self.embedding.embedding_dim == hidden_size,\
            f'hidden_size must equal embedding dim, found hidden_size={hidden_size}, embedding_dim={self.embedding.embedding_dim}'
        
        self.gru = nn.GRU(hidden_size, hidden_size)
```

### Teacher Forcing

> Consider the task of sequence prediction, so you want to predict the next element of a sequence $e_t$ given the previous elements of this sequence $e_{t−1},e_{t−2},…,e_{1}=e_{t−1:1}$. Teacher forcing is about forcing the predictions to be based on correct histories (i.e. the correct sequence of past elements) rather than predicted history (which may not be correct). To be more concrete, let $e_i$ denote the $i$th predicted element of the sequence and let $e_i$ be the corresponding ground-truth. Then, if you use teacher forcing, to predict etet, rather than using $\hat{e_{t-1:1}}$, you would use $\hat{e_{t-1:1}}$. [ai.stackexchange](https://ai.stackexchange.com/questions/18006/what-is-teacher-forcing)

Here's another explanation

> _Teacher forcing is like a teacher correcting a student as the student gets trained on a new concept. As the right input is given by the teacher to the student during training, student will learn the new concept faster and efficiently._

When training with teacher forcing, at random we choose to do forcing, in this we supply the actual output of the previous time step instead of the predicted output from the previous time step of the encoder.

```python
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
```

But why do we really have to do this?

Lets assume we have an slightly trained Network for the Encoder and Decoder

And these are our sentences

`SRC: <SOS> hi satyajit how are you ? <EOS>`
`TGT: <SOS> salut satyajit comment vas-tu ? <EOS>`

After the entire `SRC` is sent to the encoder word by word, we will have some embeddings, which would be _meaningless_ since the model is not trained that well

This is what the decoder will see

```
INPUT				PRED
[SOS]				a
[SOS] a				a ??
[SOS] a ??			a ?? ??
```

See how difficult it is for the decoder rnn to decode meaningless sentences, and this makes the model unstable and very difficult to learn, and this is why we randomly use the target sentence itself to train the decoder

```
INPUT TEACHER FORCED 		PRED
[SOS]						??
[SOS] ??					??
[SOS] ?? satyajit           ?? satyajit
[SOS] ?? satyajit how		?? satyajit comment
```

Something like above, since the decoder is fed with the actual target words as the previous input, it gets to learn better.

### Further possible improvement

- The model does not support   batching, which greatly would improve performance and also loss if done.
- The optimizer here used is SGD, which is generally not preferred for FC Networks, so Adam could have been used here.
- I found [this](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/seq2seq-translation/seq2seq-translation-batched.ipynb) really good notebook, that shows diffferent kinds of attention models, and guess what ! it is batched !


### Sample Output

```
[KEY: > input, = target, < output]

> he s not going .
= il ne s y rend pas .
< il ne s y y . <EOS>

> we re not happy .
= nous ne sommes pas heureuses .
< nous ne sommes pas heureux . <EOS>

> we re too old .
= nous sommes trop vieux .
< nous sommes trop vieux . <EOS>

> i m not a crook .
= je ne suis pas un escroc .
< je ne suis pas un . <EOS>

> you re free of all responsibility .
= vous etes liberee de toute responsabilite .
< vous etes liberee de toute responsabilite . <EOS>

> i m sorry we re completely sold out .
= je suis desole nous avons ete devalises .
< je suis desole nous avons tout vendu . <EOS>

> you are the one .
= vous etes l elu .
< vous etes celui la . <EOS>

> they re all dead .
= elles sont toutes mortes .
< ils sont tous des . <EOS>

> he s always late for school .
= il est toujours en retard a l ecole .
< il est toujours en retard a l ecole . <EOS>

> he is busy .
= il a a faire .
< il a l l l l <EOS>
```

---

### Some Attention Visualizations

```
input = i m very impressed by your work .
output = je suis tres par par votre travail . <EOS>
```
![attn1](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/attentions/attn1.png?raw=true)

---

```
input = we re smart .
output = nous sommes intelligents . <EOS>
```
![attn2](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/attentions/attn2.png?raw=true)

---
```
input = i m still hungry .
output = j ai toujours faim . <EOS>
```

![attn3](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/attentions/attn3.png?raw=true)

---
```
input = he is very eager to go there .
output = il est tres sensible de partir . <EOS>
```

![attn4](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/attentions/attn4.png?raw=true%5C)

---

```
input = i m sorry we re completely sold out .
output = je suis desole nous avons tout vendu . <EOS>
```

![attn5](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/attentions/attn5.png?raw=true)

---

<p align="center">
<iframe src="https://giphy.com/embed/dz1iM8gU3RhzQy2MC7" width="480" height="392" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/memecandy-dz1iM8gU3RhzQy2MC7"></a></p>
</p>

---
<p align="center">
Thanks for reading, have a great day 😄
</p>

<p align="center">
<iframe src="https://open.spotify.com/embed/track/3jPWd7NpYoaGVUSbJh9Xca" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
</p>

---
<p align="center">
:wq satyajit
</p>

