# 06 Encoder Decoder

## Assignment

1.  Take the last code (+tweet dataset) and convert that in such a war that:
    1.  _encoder:_ an  RNN/LSTM  layer takes the words in a sentence one by one and finally converts them into a single vector.  **VERY IMPORTANT TO MAKE THIS SINGLE VECTOR**
    2.  this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
    3.  and send this final vector to a Linear Layer and make the final prediction.
    4.  This is how it will look:
        1.  embedding
        2.  _word from a sentence +last hidden vector ->_ encoder  _-> single vector_
        3.  _single vector + last hidden vector -> decoder -> single vector_
        4.  _single vector -> FC layer -> Prediction_
2.  Your code will be checked for plagiarism, and if we find that you have copied from the internet, then -100%.
3.  The code needs to look as simple as possible, the focus is on making encoder/decoder classes and how to link objects together
4.  Getting good accuracy is NOT the target, but must achieve at least  **45%**  or more
5.  Once the model is trained, take one sentence, "print the outputs" of the encoder for each step and "print the outputs" for each step of the decoder. ←  **THIS IS THE ACTUAL ASSIGNMENT**

## Solution

| Dataset | Model |
|--|--|
| [Github](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/Tweets_Dataset.ipynb) | [Github](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/Tweets_Model.ipynb) |
| [Colab](https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/Tweets_Dataset.ipynb) | [Colab](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/Tweets_Model.ipynb) |

The Dataset consists of `1341 (Cleaned)` Tweets which are labelled `[Negative, Positive, Neutral]`

```
Negative: 931, Positive: 352, Neutral: 81
```
68% is Negative, so I need to beat at least this to claim that my model is learning

![bar too low](https://y.yarn.co/4ffcc7a9-5e92-4811-a0cb-f5fd684cff05_text.gif)


The real question though is how low of number of parameters we can go?

---


Highest Test Accuracy: `81.33%`
Epochs: `5`

Tensorboard ExperimentLogs: [https://tensorboard.dev/experiment/HaR2fvGGSn6gWznIGb645Q/#scalars](https://tensorboard.dev/experiment/HaR2fvGGSn6gWznIGb645Q/#scalars)

![val accuracy](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/val_accuracy.png?raw=true)

## Model

### The Encoder

```python
class Encoder(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=16, proj_dim=64):
        super(self.__class__, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        self.encode_lstm = nn.LSTMCell(self.input_dim, self.hidden_dim, bias=False)
        self.encoder_proj = nn.Linear(self.hidden_dim, self.proj_dim, bias=False)

    def init_hidden(self, device, batch_size):
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (zeros, zeros)
    
    def forward(self, sequences, lengths, hidden_state, debug=False):
        (hh, cc) = hidden_state

        for idx in range(lengths[0]):
            (hh, cc) = self.encode_lstm(sequences[0][idx].unsqueeze(0), (hh, cc))
            # print(hx[0][0].numpy())
            if debug:
                sns.heatmap(hh[0].detach().numpy().reshape(-1, 4), fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu").set(title=f"Encoder Hidden State, step={idx}")
                plt.show()

        encoder_sv = self.encoder_proj(hh)

        if debug:
            sns.heatmap(encoder_sv[0].detach().numpy().reshape(-1, 1), fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu").set(title=f"Encoder Single Vector")
            plt.show()

        return encoder_sv, (hh, cc)
```

### The Decoder

```python
class Decoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=16, proj_dim=64):
        super(self.__class__, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        self.decode_lstm = nn.LSTMCell(self.input_dim, self.hidden_dim, bias=False)
        self.decoder_proj = nn.Linear(self.hidden_dim, self.proj_dim, bias=False)

    def forward(self, encoder_inp, hidden_state, max_steps=5, debug=False):
        (hh, cc) = hidden_state

        for idx in range(max_steps):
            (hh, cc) = self.decode_lstm(encoder_inp, (hh, cc))
            if debug:
                sns.heatmap(hh[0].detach().numpy().reshape(-1, 4), fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu").set(title=f"Decoder Hidden State, step={idx}")
                plt.show()

        decoder_sv = self.decoder_proj(hh)
        if debug:
            sns.heatmap(decoder_sv[0].detach().numpy().reshape(-1, 1), fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu").set(title=f"Decoder Single Vector")
            plt.show()

        return decoder_sv
```

This is as simple as it gets !

## Experiments

| Encoder `[hidden_dim]`| Decoder `[hidden_dim]`   | Augmentation | Epochs | Test Accuracy | Remark |
|--|--|--|--|--|--|
| 64 | 64 | none | 5 |**81.3**| Overfit, can be better
|
| 32 | 32 | none | 5 |80.6| Reduce `hidden_dim`
|
| **16** | **16** | **none** | **5** |80.6| Reduce `hidden_dim`, Still good with just 16 dims !
|

### Conclusions and Monologue
- The accuracy was really high, even with just 16 dims, but how ? why ?
- This time I used Pretrained `GloVe` Vector `o(*￣▽￣*)ブ`
- `(￣y▽￣)╭ Ohohoho.....`
- The dataset is relatively small, so i am not so surprised, it's just 1300 tweets, which was split into 1000 train, and 300 test set, also there is class imbalance, huge, imbalance.
- The embedding was trained on 6B texts, and has a dimension of `300`, this reduces so much of the network's pain, the main guy working now is the encoder lstm, which is responsible to extract information such that the later decoder can understand what is happening and give a judgement.
- I've also created a issue in `torchtext` of how to use `Vectors`: https://github.com/pytorch/text/issues/1323

## Model Debug

Raw Tweet: `RT @WhatTheFFacts: In his teen years, Obama has been known to use marijuana and cocaine.`

Input Text: `In his teen years, Obama has been known to use marijuana and cocaine.`

Label: `Negative`

Predicted: `Negative`

The states are just beautiful to watch, look at how information is encoded !!!

<center>
<iframe src="https://giphy.com/embed/d1USMTfNFsvrG" width="480" height="333" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/cat-animal-surise-d1USMTfNFsvrG"></a></p>
</center>

### Encoder States


![encoder outputs](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/encoder_outputs.png?raw=true)

### Decoder States

![decoder states](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/decoder_outputs.png?raw=true)

### Final Proj Layer State

![proj layer state](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/06_Encoder_Decoder/final_output.png?raw=true)

## Misclassified

```
Tweet: Obama entrando en Skateboard y que si es real http://t.co/Sm9s2o9i LIKE A BOSS
Tweet Cleaned: Obama entrando en Skateboard y que si es real LIKE A BOSS
Label: Positive
Predicted: Negative


Tweet: The Writer's strike gave him the opportunity to go work for Obama's campaign
Tweet Cleaned: The Writer's strike gave him the opportunity to go work for Obama's campaign
Label: Positive
Predicted: Negative


Tweet: Great photo!! LLAP!!  #trekkers RT @GuyKawasaki: Barack Obama: Vulcan? http://t.co/UejtlmGt
Tweet Cleaned: Great photo!! LLAP!!  Barack Obama Vulcan?
Label: Positive
Predicted: Negative


Tweet: RT @marymauldin: OBAMA HONORS the wkwnd of Holy Passover 4 Jews & Holy Passion of Jesus by hosting MUSLIM BROTHERHOOD who wld murder both Jews & Christians
Tweet Cleaned:  OBAMA HONORS the wkwnd of Holy Passover 4 Jews & Holy Passion of Jesus by hosting MUSLIM BROTHERHOOD who wld murder both Jews & Christians
Label: Neutral
Predicted: Negative


Tweet: .@Notintheface1 I thought it was more, describe Obama, but instead of using Obama's qualities, list his own. Then hope nobody noticed.
Tweet Cleaned: . I thought it was more, describe Obama, but instead of using Obama's qualities, list his own. Then hope nobody noticed.
Label: Positive
Predicted: Negative


Tweet: Obama: If all the blacks line up for me, I promise I will triple your entitlements & give you all Escalades    http://t.co/f8b7HLaf
Tweet Cleaned: Obama If all the blacks line up for me, I promise I will triple your entitlements & give you all Escalades
Label: Neutral
Predicted: Positive


Tweet: RT @GatorNation41: gas was $1.92 when Obama took office...I guess he did promise he would change things http://t.co/TlfMmi0G
Tweet Cleaned:  gas was $1.92 when Obama took office...I guess he did promise he would change things
Label: Positive
Predicted: Negative


Tweet: RT @1Dlover_carrots: @Harry_Styles on a scale of 1-10 how attractive is this?...and don't say michelle Obama. http://t.co/YiFq4PKT
Tweet Cleaned:  on a scale of 1-10 how attractive is this?...and don't say michelle Obama.
Label: Positive
Predicted: Negative


Tweet: Liberal #SteveJobs: Obama's business killing regulations forces Apple to build in #China #cnn - http://t.co/Yd8jzwoV
Tweet Cleaned: Liberal  Obama's business killing regulations forces Apple to build in -
Label: Positive
Predicted: Negative


Tweet: Saul says Pres Obama "will do anything" to distract Americans from his "failed" economic record incl unemployment & higher gas prices."
Tweet Cleaned: Saul says Pres Obama "will do anything" to distract Americans from his "failed" economic record incl unemployment & higher gas prices."
Label: Positive
Predicted: Neutral
```

## Correct Classified

```
Tweet: RT @WhatTheFFacts: In his teen years, Obama has been known to use marijuana and cocaine.
Tweet Cleaned:  In his teen years, Obama has been known to use marijuana and cocaine.
Label: Negative
Predicted: Negative


Tweet: RT @Drudge_Report: Obama setting up Supreme Court as campaign issue... http://t.co/1IiLN01H
Tweet Cleaned:  Obama setting up Supreme Court as campaign issue...
Label: Positive
Predicted: Positive


Tweet: RT @NatlWOW: @edshow Pres. Obama understands right from wrong! And doesn't need to flip flop around to get votes! #UniteWomen #edshow
Tweet Cleaned:  Pres. Obama understands right from wrong! And doesn't need to flip flop around to get votes!
Label: Negative
Predicted: Negative


Tweet: #WhatsRomneyHiding HE WONDERING.. WHATS OBAMA HIDING?????  remember the most transparent adm in history.. LMBO
Tweet Cleaned: HE WONDERING.. WHATS OBAMA HIDING????? remember the most transparent adm in history.. LMBO
Label: Negative
Predicted: Negative


Tweet: President Obama * Lindsay Lohan * 1989 RUMORS business 19 TH & M ST NW DC met field agent = multi connector to FFX  VA covert overt zone.
Tweet Cleaned: President Obama * Lindsay Lohan * 1989 RUMORS business 19 TH & M ST NW DC met field agent = multi connector to FFX VA covert overt zone.
Label: Negative
Predicted: Negative


Tweet: Romney and Obama agree that Augusta National should allow women to be members? Unthinkable...and bad news for the green coats.
Tweet Cleaned: Romney and Obama agree that Augusta National should allow women to be members? Unthinkable...and bad news for the green coats.
Label: Negative
Predicted: Negative


Tweet: #WhatsRomneyHiding Obama released his tax returns since 2000, where are Romney's?
Tweet Cleaned: Obama released his tax returns since 2000, where are Romney's?
Label: Negative
Predicted: Negative


Tweet: #newbedon 4/6/2012 4:25:20 AM Obama Wins Landslide Presidential Election...With Online Gamers http://t.co/JGbJwE9Z
Tweet Cleaned: 4/6/2012 42520 AM Obama Wins Landslide Presidential Election...With Online Gamers
Label: Negative
Predicted: Negative


Tweet: Obama says knock you out -- http://t.co/PUZRq7HU #screwytees
Tweet Cleaned: Obama says knock you out --
Label: Negative
Predicted: Negative


Tweet: Top Secret Obama 2012 World War 3 Illuminati Antichrist Conspiracy!: http://t.co/iqg1xarL via @youtube
Tweet Cleaned: Top Secret Obama 2012 World War 3 Illuminati Antichrist Conspiracy! via
Label: Negative
Predicted: Negative
```

## That's it Folks !

<center>
<iframe src="https://giphy.com/embed/GypVyX5Nw0R2g" width="480" height="362" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/cat-funny-GypVyX5Nw0R2g"></a></p>
</center>
