# 11 Attention Advanced

## Assignment

1.  Follow the similar strategy as we did in our  [baby-steps-code (Links to an external site.)](https://colab.research.google.com/drive/1IlorkvXhZgmd_sayOVx4bC_I5Qpdzxk_?usp=sharing), but replace GRU with LSTM. In your code you must:
    1.  Perform 1 full feed forward step for the encoder **manually**
    2.  Perform 1 full feed forward step for the decoder **manually**.
    3.  You can use any of the 3 attention mechanisms that we discussed.
2.  Explain your steps in the readme file and
3.  Submit the assignment asking for these things:
    1.  Link to the readme file that must explain Encoder/Decoder Feed-forward manual steps  **and the attention mechanism that you have used** - 500 pts
    2.  Copy-paste (don't redirect to github), the Encoder Feed Forward steps for 2 words - 250 pts
    3.  Copy-paste (don't redirect to github), the Decoder Feed Forward steps for 2 words - 250 pts


## Solution

|| NBViewer | Google Colab | Tensorboard Logs
|--|--|--|--|
| Attention Advanced - **Solution** | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/11_Attention_Advanced/Attention_Advanced.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/11_Attention_Advanced/Attention_Advanced.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> | <a href="https://tensorboard.dev/experiment/kwH5WKoQTOaJLhOc9oYy6Q/"><img src="https://img.shields.io/badge/logs-tensorboard-orange?logo=Tensorflow"></a> |
| Seq2Seq-Attention (Reference) | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/11_Attention_Advanced/seq2seq-translation.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/11_Attention_Advanced/seq2seq-translation.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> | |


If someday PyTorch decides to remove the `data.zip` file, I've added it to [this repository](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/10_Seq2Seq_Attention/data.zip).

### Encoder Feed Forward Steps



```python
enc_embedding = nn.Embedding(
    hparams.input_dim,
    hparams.hidden_size
)
enc_embedding
```




    >> Embedding(4347, 64)




```python
enc_lstm = nn.LSTM(
    hparams.hidden_size,
    hparams.hidden_size,
    num_layers=1
)
enc_lstm
```




    >> LSTM(64, 64)





```python
f'The encoder will take the input sentence {src_text[0]} = {" ".join(input_lang_itos[x] for x in src_text[0])}'
```




    >> 'The encoder will take the input sentence tensor([ 13,  16, 463,   4,   3]) = tu es impatiente . <eos>'




```python
enc_encoder_hidden = torch.zeros(1, 1, hparams.hidden_size), torch.zeros(1, 1, hparams.hidden_size)
enc_encoder_hidden[0].shape, enc_encoder_hidden[1].shape
```




    >> (torch.Size([1, 1, 64]), torch.Size([1, 1, 64]))




```python
# forward pass
seq_len = len(src_text[0])
embedded = enc_embedding(src_text[0]).view(seq_len, 1, -1)
enc_output, enc_hidden = enc_lstm(embedded, enc_encoder_hidden)
```

Note `enc_output` and `enc_hidden` will later be used by the decoder !


```python
print(
    f'since our sentence has {len(src_text[0])} words, the number of tensors in enc_output is {enc_output.shape[0]}'
)
```

    >> since our sentence has 5 words, the number of tensors in enc_output is 5



```python
print(
    f'embedding: {embedded.shape}\nenc_output: {enc_output.shape}\nenc_hidden: {enc_hidden[0].shape, enc_hidden[1].shape}'
)
```

    >> embedding: torch.Size([5, 1, 64])
    >> enc_output: torch.Size([5, 1, 64])
    >> enc_hidden: (torch.Size([1, 1, 64]), torch.Size([1, 1, 64]))



```python
print(
    f'encoding for the word \'{input_lang_itos[src_text[0, 0]]}\' =>\n\n{enc_output[0]}'
)
```

    encoding for the word 'tu' =>
    
    tensor([[-0.0543,  0.0638, -0.1975, -0.1677, -0.0539, -0.0799, -0.0503,  0.0007,
             -0.1526,  0.0269,  0.1492,  0.2136,  0.0022, -0.0716,  0.0493,  0.0884,
              0.2390, -0.1747,  0.0222,  0.1018,  0.0792, -0.1830,  0.2660, -0.1601,
             -0.0031,  0.2112,  0.1274, -0.2266,  0.1665, -0.0918,  0.1431, -0.1941,
              0.1174, -0.1755,  0.2341, -0.1604, -0.0336, -0.0107, -0.0823,  0.2096,
             -0.1492,  0.0024, -0.2048, -0.2197, -0.0225, -0.0126,  0.1423, -0.0376,
              0.0351, -0.0735,  0.1298,  0.0437,  0.1812, -0.1990, -0.0230, -0.1988,
             -0.0519, -0.0607, -0.0144,  0.0720, -0.2157, -0.0570,  0.0637, -0.0687]],
           grad_fn=<SelectBackward>)



```python
print(
    f'encoding for the word \'{input_lang_itos[src_text[0, 1]]}\' =>\n\n{enc_output[1]}'
)
```

    encoding for the word 'es' =>
    
    tensor([[ 0.0052, -0.1731, -0.2374, -0.0742, -0.1043, -0.0528, -0.0789,  0.0948,
             -0.1553, -0.1581,  0.3061,  0.0233, -0.0263, -0.1190,  0.2288,  0.3443,
              0.3691, -0.0884, -0.1495,  0.0013, -0.0716,  0.0549,  0.2131, -0.0852,
             -0.1066,  0.2260,  0.0553, -0.0925,  0.2297, -0.0972,  0.2397, -0.0222,
              0.0623, -0.3111,  0.2283, -0.1766,  0.0787, -0.0744, -0.0616,  0.0231,
             -0.0838,  0.0849, -0.4161,  0.0202, -0.1292, -0.0138,  0.0784,  0.0334,
             -0.1377, -0.0678,  0.0150,  0.1796, -0.0396,  0.1356,  0.0032, -0.0631,
              0.0989,  0.0350, -0.2848, -0.0165, -0.1672, -0.0179,  0.1378,  0.0915]],
           grad_fn=<SelectBackward>)



```python
print(
    f'encoding for the word \'{input_lang_itos[src_text[0, 2]]}\' =>\n\n{enc_output[2]}'
)
```

    encoding for the word 'impatiente' =>
    
    tensor([[ 0.0732, -0.1036, -0.1111, -0.0589, -0.0526, -0.0549, -0.1882,  0.2312,
              0.0089, -0.1297, -0.0507, -0.1168, -0.0353,  0.0075,  0.0738,  0.1092,
              0.0566,  0.0912, -0.2511, -0.0793, -0.0123,  0.1230,  0.1665, -0.0725,
             -0.0762,  0.1660, -0.1273,  0.0871,  0.2491,  0.0638,  0.2093,  0.0893,
              0.1419, -0.2067,  0.0119, -0.1068,  0.1178,  0.0655, -0.0208, -0.0647,
             -0.1147, -0.0500, -0.0150, -0.0616, -0.2934, -0.1099, -0.2117,  0.1308,
             -0.1164, -0.0382,  0.1587,  0.1247, -0.1988,  0.1069,  0.0867, -0.0014,
              0.0079, -0.0263, -0.1672,  0.0169, -0.0829,  0.0871,  0.0611,  0.0820]],
           grad_fn=<SelectBackward>)


and so on every word is now encoded

### Decoder Feed Forward Steps

**A complete step by step Decoder Feed Forward** 

The Decoder parameters


```python
dec_embedding = nn.Embedding(
    hparams.output_dim,
    hparams.hidden_size
)
dec_embedding
```




    >> Embedding(2805, 64)




```python
dec_lstm = nn.LSTM(
    hparams.hidden_size * 2,
    hparams.hidden_size, 1,
)
dec_lstm
```




    >> LSTM(128, 64)




```python
dec_out = nn.Linear(
    hparams.hidden_size * 2, hparams.output_dim
)
dec_out
```




    >> Linear(in_features=128, out_features=2805, bias=True)



This is the attention part


```python
luong_attn = nn.Linear(
    hparams.hidden_size, hparams.hidden_size
)
luong_attn
```




    >> Linear(in_features=64, out_features=64, bias=True)




```python
from ttctext.datamodules.torch_translate import SOS_token, EOS_token, PAD_token
```


```python
decoder_input = torch.tensor([SOS_token]).unsqueeze(0) # SOS is the first word to the decoder
decoder_context = torch.zeros(1, hparams.hidden_size)
decoder_hidden = enc_encoder_hidden # Use last hidden state from encoder to start decoder
```


```python
print(
    f'decoder_input:\t\t{decoder_input.shape}\ndecoder_context:\t{decoder_context.shape}\ndecoder_hidden:\t\t{decoder_hidden[0].shape, decoder_hidden[1].shape}'
)
```

    >> decoder_input:		torch.Size([1, 1])
    >> decoder_context:	torch.Size([1, 64])
    >> decoder_hidden:		(torch.Size([1, 1, 64]), torch.Size([1, 1, 64]))



```python
dec_word_input = decoder_input

# Get the embedding of the current input word (last output word)
dec_word_embedded = dec_embedding(dec_word_input).view(1, 1, -1) # S=1 x B x N
f'dec_word_embedded: {dec_word_embedded.shape}'
```




    >> 'dec_word_embedded: torch.Size([1, 1, 64])'




```python
last_context = decoder_context
last_hidden = decoder_hidden

# Combine embedded input word and last context, run through RNN
dec_rnn_input = torch.cat((dec_word_embedded, last_context.unsqueeze(0)), dim=2)
dec_rnn_output, dec_rnn_hidden = dec_lstm(dec_rnn_input, last_hidden)
```


```python
print(
    f'dec_rnn_input:\t\t{dec_rnn_input.shape}\ndec_rnn_output:\t\t{dec_rnn_output.shape}\ndec_rnn_hidden:\t\t{dec_rnn_hidden[0].shape, dec_rnn_hidden[0].shape}'
)
```

    >> dec_rnn_input:		torch.Size([1, 1, 128])
    >> dec_rnn_output:		torch.Size([1, 1, 64])
    >> dec_rnn_hidden:		(torch.Size([1, 1, 64]), torch.Size([1, 1, 64]))



```python
print(
    f'dec_rnn_output: {dec_rnn_output.shape} value =>\n\n{dec_rnn_output}'
)
```

    dec_rnn_output: torch.Size([1, 1, 64]) value =>
    
    tensor([[[ 2.4784e-02,  1.3236e-01,  2.8387e-02,  3.1887e-03,  1.3423e-01,
              -8.7927e-02,  1.5511e-01,  2.8784e-02,  1.0580e-01, -1.4575e-01,
              -4.3383e-03, -5.2812e-02,  1.7754e-01, -3.1593e-02,  6.7075e-02,
               7.7494e-02,  4.9320e-02,  1.7713e-01, -2.0790e-01, -5.0475e-02,
              -5.8649e-02,  4.6692e-02, -3.1964e-02, -1.3329e-01, -9.9950e-02,
              -9.4949e-02,  2.1983e-02,  1.2766e-01,  3.3407e-02,  1.2375e-02,
              -9.7627e-02,  8.2564e-05,  4.0589e-02, -5.0377e-02, -6.5772e-02,
               2.8655e-02, -1.1418e-01,  5.5525e-02,  1.6390e-01,  3.2977e-02,
               1.3898e-02,  4.4744e-02, -5.5388e-02, -5.9081e-02,  2.2567e-02,
              -1.6297e-01,  9.2167e-02, -1.4468e-01,  5.4815e-02, -8.2351e-02,
              -4.3525e-02,  6.9047e-02,  5.4019e-02, -1.1496e-01, -1.9732e-01,
               5.2014e-02,  2.2706e-01,  6.1765e-02,  1.0778e-01,  7.5064e-02,
               1.1164e-01, -4.1908e-02, -7.5117e-02,  1.3715e-02]]],
           grad_fn=<StackBackward>)


Calculate attention from current RNN state


```python
seq_len = len(enc_output)

# Create variable to store attention energies
attn_energies = torch.zeros(seq_len) # B x 1 x S

attn_hidden = dec_rnn_output.squeeze(0)

# Calculate energies for each encoder output
for i in range(seq_len):
    energy = luong_attn(enc_output[i])
    energy = attn_hidden.view(-1).dot(energy.view(-1))
    
    print(f'energy for {i}th word: {energy}')

    attn_energies[i] = energy
```

    >> energy for 0th word: 0.013496480882167816
    >> energy for 1th word: -0.019721370190382004
    >> energy for 2th word: 0.04026033729314804
    >> energy for 3th word: 0.11657365411520004
    >> energy for 4th word: 0.05412563309073448



```python
print(
    f'attention energies: {attn_energies.shape} values =>\n\n{attn_energies}',
)
```

    attention energies: torch.Size([5]) values =>
    
    tensor([ 0.0135, -0.0197,  0.0403,  0.1166,  0.0541], grad_fn=<CopySlices>)



```python
# Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
dec_attn_weights =  F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)
```


```python
dec_attn_weights.sum()
```




    >> tensor(1., grad_fn=<SumBackward0>)




```python
print(
    f'attention weights: {dec_attn_weights.shape} values =>\n\n{dec_attn_weights}'
)
```

    attention weights: torch.Size([1, 1, 5]) values =>
    
    >> tensor([[[0.1944, 0.1880, 0.1997, 0.2155, 0.2024]]],
           grad_fn=<UnsqueezeBackward0>)



```python
# apply to encoder outputs
dec_context_new = dec_attn_weights.bmm(enc_output.transpose(0, 1)) # B x 1 x N
```


```python
print(
    f'dec_context_new: {dec_context_new.shape} values =>\n\n{dec_context_new}'
)
```

    dec_context_new: torch.Size([1, 1, 64]) values =>
    
    tensor([[[ 0.0873, -0.0190, -0.0559, -0.0386, -0.0353,  0.0400, -0.0116,
               0.1641, -0.0525, -0.0474,  0.0651,  0.0297, -0.0100, -0.1076,
               0.0932,  0.1184,  0.1446, -0.0358, -0.2018,  0.0410, -0.0010,
              -0.0116,  0.1718, -0.0905, -0.0948,  0.2211, -0.1088, -0.0856,
               0.0962, -0.0698,  0.0966, -0.0301,  0.1399, -0.1744,  0.0716,
              -0.1258,  0.0235, -0.0409, -0.0293,  0.0687, -0.0779,  0.0336,
              -0.1106,  0.0323, -0.1402, -0.0324, -0.0821,  0.1166, -0.1192,
               0.0371,  0.1582,  0.0321, -0.1342,  0.1206,  0.0797, -0.0553,
              -0.0261,  0.0474, -0.0550,  0.0025, -0.0787, -0.0014, -0.0070,
               0.0421]]], grad_fn=<BmmBackward0>)



```python
# Final output layer (next word prediction) using the RNN hidden state and context vector
dec_rnn_output_new = dec_rnn_output.squeeze(0) # S=1 x B x N -> B x N
dec_context_new = dec_context_new.squeeze(1)       # B x S=1 x N -> B x N
dec_output_final = F.log_softmax(
    dec_out(torch.cat((dec_rnn_output_new, dec_context_new), dim=1)),
    dim=-1
)
```


```python
torch.log(F.softmax(
    dec_out(torch.cat((dec_rnn_output_new, dec_context_new), dim=1)),
    dim=-1
)).sum()
```




    >> tensor(-22277.1465, grad_fn=<SumBackward0>)




```python
dec_output_final.sum()
```




    >> tensor(-22277.1465, grad_fn=<SumBackward0>)




```python
print(
    f'size after concatenating dec_context and dec_rnn_output: {torch.cat((dec_rnn_output_new, dec_context_new), dim=1).shape}'
)
```

    size after concatenating dec_context and dec_rnn_output: torch.Size([1, 128])


After applying the final FC layer of decoder output


```python
print(
    f'dec_rnn_output_new: {dec_rnn_output_new.shape} values =>\n\n{dec_rnn_output_new}'
)
```

    dec_rnn_output_new: torch.Size([1, 64]) values =>
    
    tensor([[ 2.4784e-02,  1.3236e-01,  2.8387e-02,  3.1887e-03,  1.3423e-01,
             -8.7927e-02,  1.5511e-01,  2.8784e-02,  1.0580e-01, -1.4575e-01,
             -4.3383e-03, -5.2812e-02,  1.7754e-01, -3.1593e-02,  6.7075e-02,
              7.7494e-02,  4.9320e-02,  1.7713e-01, -2.0790e-01, -5.0475e-02,
             -5.8649e-02,  4.6692e-02, -3.1964e-02, -1.3329e-01, -9.9950e-02,
             -9.4949e-02,  2.1983e-02,  1.2766e-01,  3.3407e-02,  1.2375e-02,
             -9.7627e-02,  8.2564e-05,  4.0589e-02, -5.0377e-02, -6.5772e-02,
              2.8655e-02, -1.1418e-01,  5.5525e-02,  1.6390e-01,  3.2977e-02,
              1.3898e-02,  4.4744e-02, -5.5388e-02, -5.9081e-02,  2.2567e-02,
             -1.6297e-01,  9.2167e-02, -1.4468e-01,  5.4815e-02, -8.2351e-02,
             -4.3525e-02,  6.9047e-02,  5.4019e-02, -1.1496e-01, -1.9732e-01,
              5.2014e-02,  2.2706e-01,  6.1765e-02,  1.0778e-01,  7.5064e-02,
              1.1164e-01, -4.1908e-02, -7.5117e-02,  1.3715e-02]],
           grad_fn=<SqueezeBackward1>)



```python
print(
    f'dec_context_new: {dec_context_new.shape} values =>\n\n{dec_context_new}'
)
```

    dec_context_new: torch.Size([1, 64]) values =>
    
    tensor([[ 0.0873, -0.0190, -0.0559, -0.0386, -0.0353,  0.0400, -0.0116,  0.1641,
             -0.0525, -0.0474,  0.0651,  0.0297, -0.0100, -0.1076,  0.0932,  0.1184,
              0.1446, -0.0358, -0.2018,  0.0410, -0.0010, -0.0116,  0.1718, -0.0905,
             -0.0948,  0.2211, -0.1088, -0.0856,  0.0962, -0.0698,  0.0966, -0.0301,
              0.1399, -0.1744,  0.0716, -0.1258,  0.0235, -0.0409, -0.0293,  0.0687,
             -0.0779,  0.0336, -0.1106,  0.0323, -0.1402, -0.0324, -0.0821,  0.1166,
             -0.1192,  0.0371,  0.1582,  0.0321, -0.1342,  0.1206,  0.0797, -0.0553,
             -0.0261,  0.0474, -0.0550,  0.0025, -0.0787, -0.0014, -0.0070,  0.0421]],
           grad_fn=<SqueezeBackward1>)



```python
print(
    f'dec_output_final: {dec_output_final.shape} values =>\n\n{dec_output_final}'
)
```

    dec_output_final: torch.Size([1, 2805]) values =>
    
    >> tensor([[-7.8912, -7.8641, -7.9288,  ..., -7.8772, -7.9572, -8.1186]],
           grad_fn=<LogSoftmaxBackward>)



```python
dec_topv, dec_topi = dec_output_final.data.topk(1)
ni = dec_topi[0, 0]
```


```python
dec_topv, dec_topi
```




    >> (tensor([[-7.6851]]), tensor([[2579]]))


```python
print(
    f'predicted word: {target_lang_itos[ni]}'
)
```

    >> predicted word: samples


**Now we run it for two inputs**

```python
decoder_input = torch.tensor([SOS_token]).unsqueeze(0) # SOS is the first word to the decoder
decoder_context = torch.zeros(1, hparams.hidden_size)
decoder_hidden_test = enc_encoder_hidden # Use last hidden state from encoder to start decoder
```


```python
last_context = decoder_context
last_hidden = decoder_hidden
```


```python
i = 0

dec_word_input = decoder_input

print(f'decoder word input: {dec_word_input} value: {target_lang_itos[dec_word_input[0, 0]]}')

dec_word_embedded = dec_embedding(dec_word_input).view(1, 1, -1)

dec_rnn_input = torch.cat((dec_word_embedded, last_context.unsqueeze(0)), dim=2)
dec_rnn_output, dec_rnn_hidden_new = dec_lstm(dec_rnn_input, last_hidden)

# --- attn
seq_len = len(enc_output)
attn_energies = torch.zeros(seq_len)
attn_hidden = dec_rnn_output.squeeze(0).squeeze(0)
for i in range(seq_len):
    energy = luong_attn(enc_output[i]).squeeze(0)
    energy = attn_hidden.dot(energy)
    attn_energies[i] = energy
# --- attn

dec_attn_weights = F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)

print(f'attentions: {dec_attn_weights[0, 0].detach().numpy()}')

dec_context_new = dec_attn_weights.bmm(enc_output.transpose(0, 1))

dec_rnn_output_new = dec_rnn_output.squeeze(0) # S=1 x B x N -> B x N
dec_context_new = dec_context_new.squeeze(1)       # B x S=1 x N -> B x N
dec_output_final = F.log_softmax(
    dec_out(torch.cat((dec_rnn_output_new, dec_context_new), 1)),
    dim=-1
)

dec_topv, dec_topi = dec_output_final.data.topk(1)
ni = dec_topi[0, 0]

print(
    f'predicted word: {target_lang_itos[ni]}'
)

decoder_input = torch.tensor([ni]).unsqueeze(0)

last_context = dec_context_new
last_hidden = dec_rnn_hidden_new

print('\n')
```

    >> decoder word input: tensor([[2]]) value: <sos>
    >> attentions: [0.1943825  0.18803161 0.19965519 0.21548797 0.20244274]
    >> predicted word: samples
    
    



```python
last_context = decoder_context
last_hidden = decoder_hidden
```


```python
i = 1

dec_word_input = decoder_input

print(f'decoder word input: {dec_word_input} value: {target_lang_itos[dec_word_input[0, 0]]}')

dec_word_embedded = dec_embedding(dec_word_input).view(1, 1, -1)

dec_rnn_input = torch.cat((dec_word_embedded, last_context.unsqueeze(0)), dim=2)
dec_rnn_output, dec_rnn_hidden_new = dec_lstm(dec_rnn_input, last_hidden)

# --- attn
seq_len = len(enc_output)
attn_energies = torch.zeros(seq_len)
attn_hidden = dec_rnn_output.squeeze(0).squeeze(0)
for i in range(seq_len):
    energy = luong_attn(enc_output[i]).squeeze(0)
    energy = attn_hidden.dot(energy)
    attn_energies[i] = energy
# --- attn

dec_attn_weights = F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)

print(f'attentions: {dec_attn_weights[0, 0].detach().numpy()}')

dec_context_new = dec_attn_weights.bmm(enc_output.transpose(0, 1))

dec_rnn_output_new = dec_rnn_output.squeeze(0) # S=1 x B x N -> B x N
dec_context_new = dec_context_new.squeeze(1)       # B x S=1 x N -> B x N
dec_output_final = F.log_softmax(
    dec_out(torch.cat((dec_rnn_output_new, dec_context_new), 1)),
    dim=-1
)

dec_topv, dec_topi = dec_output_final.data.topk(1)
ni = dec_topi[0, 0]

print(
    f'predicted word: {target_lang_itos[ni]}'
)

decoder_input = torch.tensor([ni]).unsqueeze(0)

last_context = dec_context_new
last_hidden = dec_rnn_hidden_new

print('\n')
```

    >> decoder word input: tensor([[2579]]) value: samples
    >> attentions: [0.21555178 0.19511788 0.20333609 0.19306885 0.19292536]
    >> predicted word: shot
    
    
For even more verbosity please look into the notebook file for this assignment. (links are to the top)

## Evaluation

```text
[KEY: > input, = target, < output]

> elle est de mauvaise humeur .
= she is in a bad mood .
< she is in a mood . <EOS>

> je suis dure a cuire .
= i m tough .
< i m tough . <EOS>

> j etudie l economie a l universite .
= i m studying economics at university .
< i m studying the college . <EOS>

> je n en suis pas trop convaincu .
= i m not too convinced .
< i m not too too . <EOS>

> je suis ravie de t aider .
= i am glad to help you .
< i m glad to help you . <EOS>

> elle a tres peur des chiens .
= she s very afraid of dogs .
< she is very afraid of dogs . <EOS>

> il est fier d etre musicien .
= he is proud of being a musician .
< he s proud of a of . <EOS>

> c est le portrait crache de son pere .
= he is the image of his father .
< he is the for father father . <EOS>

> je suis juste paresseuse .
= i m just lazy .
< i m just . . <EOS>

> nous sommes en train de nous en charger .
= we re handling it .
< we re back . <EOS>
```
