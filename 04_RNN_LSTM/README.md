# RNN & LSTM

## Assignment

Assignment: 

- Refer to the file we wrote in the [class](https://colab.research.google.com/drive/1-xwX32O0WYOqcCROJnnJiSdzScPCudAM?usp=sharing): Rewrite this code, but this time remove RNN and add LSTM. 

- Refer to this [file](https://colab.research.google.com/drive/12Pciev6dvYBJ7KxwSHruG-XMwcoj0SfJ). 

  - The questions this time are already mentioned in the file. Take as much time as you want (but less than 7 days), to solve the file. Once you are done, then write your solutions in the quiz. 

  - Please note that the Session 4 Assignment Solution will time out after 15 minutes, as you just have to copy-paste your answers. 

## Solution

|Notebook Title|Colab Link| GitHub Link|
|---|---|---|
|LSTM Rewrite of the RNN Code| [Google Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/04_LSTM_IMDB.ipynb) | [GitHub](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/04_LSTM_IMDB.ipynb) |
| Original RNN Code | [Google Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/END2_Session_4.ipynb) | [GitHub](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/END2_Session_4.ipynb) |
Quiz Solution | [Google Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/04_LSTM_Practice.ipynb) | [GitHub](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/04_LSTM_Practice.ipynb) |
Quiz Question | [Google Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/EVA_P2S3.ipynb) | [GitHub](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/EVA_P2S3.ipynb) |

😢  I messed up the Quiz Again, forgot to set the hidden size to 100 for the above cells, only did it for the last training cells.

I also wrote a PyTorch Lightning version of the IMDB Text Classification: [GoogleColab](https://githubtocolab.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/IMDB_PyTorch_Lightning.ipynb) [GitHub](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/04_RNN_LSTM/IMDB_PyTorch_Lightning.ipynb)

NOTES:

- The PyTorch Text codes shared above is really outdated, i need to follow up on [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html) and rewrite the code for my sanity.
- Captum seems like a nice addition to PyTorch Text, more here: [https://captum.ai/tutorials/IMDB_TorchText_Interpret](https://captum.ai/tutorials/IMDB_TorchText_Interpret)
