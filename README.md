<h1 align="center">TSAI-DeepNLP-END2.0</h1>

---

<div align="center">
<img src="logo.png" >
</div>

---

## Website: [https://extensive-nlp.github.io/TSAI-DeepNLP-END2.0/](https://extensive-nlp.github.io/TSAI-DeepNLP-END2.0/)

Best viewed on the website

NOTE: There might be dependencies issues with torchtext version used in the notebooks, please see [this](DEPENDENCIES.md) which may solve the issue

---
<div align="center">

<img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsatyajitghana%2FTSAI-DeepNLP-END2.0&count_bg=%2379C83D&title_bg=%23555555&icon=pytorch.svg&icon_color=%23E7E7E7&title=satyajitghana%20traffic&edge_flat=false" />

<img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fextensive-nlp%2FTSAI-DeepNLP-END2.0&count_bg=%2379C83D&title_bg=%23555555&icon=pytorch.svg&icon_color=%23E7E7E7&title=extensive-nlp%20traffic&edge_flat=false" />

</div>

---

1. [Very Basics](01_VeryBasics)

    This describes all the basics of a neural network, how gradient descent works, learning rate, fully connected neurons, chain rule, etc.

2. [BackProp](02_BackProp/README.html)

    We built a Neural Network in a damn spreadsheet :)

3. [PyTorch 101](03_PyTorch101/README.html)

    Basics of PyTorch. Here i built a custom MNIST model, that can classify MNIST Image as well as do addition of that predicted image with a random integer.

4. [RNN & LSTMS](04_RNN_LSTM/README.md)

    Built an LSTM From Scratch and Trained an IMDb Sentiment analysis classifier using RNN & LSTM with PyTorch Text.

5. [LSTM & NLP Augmentation](05_NLP_Augment/README.html)

    Trained a LSTM Model on the SST Dataset. And did a lot of NLP Augmentations.

6. [Encoder Decoder](06_Encoder_Decoder/index.html)

    A Simple Encoder-Decoder Architecture, but for Classification ! I got to learn how encoder-decoder's work, and how the feature vector is used to compress and extract information :)

7. [Seq2Seq](07_Seq2Seq/index.html)

    Simple Sequence-to-Sequence Model for Question Answer and Similar Question Generation

    Also includes a redo of `5` without any augmentation. [05 Redo](07_Seq2Seq/07_SST_Redo.html)

8. [TorchText](08_TorchText/index.html)

    Introduction to the new TorchText APIs, and deprecation of `torchtext.legacy`

    Here we convert few notebooks with legacy code to the modern torchtext `0.9.0+`
