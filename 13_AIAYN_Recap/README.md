
# 13 AIAYN Recap

## Assignment

This  [code (Links to an external site.)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb#scrollTo=FqXbPB80r8p4)is from the same repo that we were following.

Your assignment is to remove all the legacy stuff from this and submit:

1.  Last 5 Training EPOCH logs
2.  Sample translation for 10 example
3.  Link to your repo

## Solution

|| NBViewer | Google Colab |
|--|--|--|
| Attention Is All You Need  | <a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/13_AIAYN_Recap/Attention_is_All_You_Need_Modern.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/13_AIAYN_Recap/Attention_is_All_You_Need_Modern.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> 

**Logs**

```text
Epoch: 01 | Time: 0m 22s
	Train Loss: 4.484 | Train PPL:  88.565
	 Val. Loss: 3.308 |  Val. PPL:  27.322
Epoch: 02 | Time: 0m 22s
	Train Loss: 2.898 | Train PPL:  18.142
	 Val. Loss: 2.602 |  Val. PPL:  13.485
Epoch: 03 | Time: 0m 23s
	Train Loss: 2.273 | Train PPL:   9.711
	 Val. Loss: 2.297 |  Val. PPL:   9.944
Epoch: 04 | Time: 0m 22s
	Train Loss: 1.884 | Train PPL:   6.580
	 Val. Loss: 2.146 |  Val. PPL:   8.555
Epoch: 05 | Time: 0m 22s
	Train Loss: 1.611 | Train PPL:   5.008
	 Val. Loss: 2.080 |  Val. PPL:   8.005
Epoch: 06 | Time: 0m 23s
	Train Loss: 1.389 | Train PPL:   4.011
	 Val. Loss: 2.052 |  Val. PPL:   7.781
Epoch: 07 | Time: 0m 22s
	Train Loss: 1.215 | Train PPL:   3.370
	 Val. Loss: 2.058 |  Val. PPL:   7.831
Epoch: 08 | Time: 0m 23s
	Train Loss: 1.062 | Train PPL:   2.892
	 Val. Loss: 2.083 |  Val. PPL:   8.030
Epoch: 09 | Time: 0m 23s
	Train Loss: 0.932 | Train PPL:   2.538
	 Val. Loss: 2.092 |  Val. PPL:   8.103
Epoch: 10 | Time: 0m 23s
	Train Loss: 0.821 | Train PPL:   2.273
	 Val. Loss: 2.103 |  Val. PPL:   8.195
```

**Sample Translations**

Random Translations from Test Set

```text
src = 2 Blonde Mädchen sitzen auf einem Sims auf einem belebten Platz.
pred = 2 blond - haired blond - haired lady sitting on a busy afternoon .
trg = 2 blond girls are sitting on a ledge in a crowded plaza.

src = Ein Mann beugt sich zur Seite und zieht etwas aus einer Tasche.
pred = An alley about to fix a very spinning fixing something from a bull .
trg = A man is leaning over and pulling something out of a bag.

src = Eine Blondine hält mit einem Mann im Sand Händchen.
pred = A very football player holding a turn to fall .
trg = A blond holding hands with a guy in the sand.

src = Der Teenager springt mit seinem Fahrrad über den Hügel.
pred = The professional football player is jumping over his turn to fall .
trg = The teen jumps the hill with his bicycle.

src = Eine Gruppe von Leuten sitzt im Freien um einen kleinen, kurzen Tisch.
pred = In a roller derby , a very eight , who is sitting around a little boy who is being pulled out .
trg = A group of people sit outdoors around a small, short table.

src = Menschen an der Seitenlinie bei einem Fußballspiel.
pred = Runners competing in a race being moved .
trg = People on the sideline of a soccer match.

src = Eine Frau spielt im Park mit zwei kleinen Jungen.
pred = A very lacrosse game is playing with two young boys in the mud .
trg = A woman playing with two young boys at a park

src = Eine Straße neben einem interessanten Ort mit vielen Säulen.
pred = A very racing cart next to a dirt pipe .
trg = A road next to an interesting place with lots of pillars.

src = Der gelbe Hund trägt einen Stock übers Wasser.
pred = The yellow bowling driver is carrying a bowling ball .
trg = The yellow dog is carrying a stick by water.

src = Eine Frau hält eine kleine weiße Statue.
pred = Firefighters holding a white pipe
trg = A woman is holding a small white statue.
```


**Attention Visualization**

A Sample from Training Set

```python
src = 'Eine Frau mit einer großen Geldbörse geht an einem Tor vorbei.'
trg = 'A woman with a large purse is walking by a gate.'

predicted trg = ['Firefighters', 'with', 'a', 'large', 'bowl', 'walking', 'past', 'a', 'hose', '.', '<eos>']
```

![train set attn](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/13_AIAYN_Recap/attns/attn_1.png?raw=true)

---

A Sample from Validation Test

```python
src = 'Ein brauner Hund rennt dem schwarzen Hund hinterher.'
trg = 'A brown dog is running after the black dog.'

predicted trg = ['A', 'brown', 'race', 'is', 'running', 'after', 'a', 'black', 'softball', '.', '<eos>']
```


![validation set attn](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/13_AIAYN_Recap/attns/attn_2.png?raw=true)

---

A Sample from Test Set

```python
src = 'Eine Mutter und ihr kleiner Sohn genießen einen schönen Tag im Freien.'
trg = 'A mother and her young song enjoying a beautiful day outside.'

predicted trg = ['In', 'the', 'middle', 'and', 'a', 'small', 'race', ',', 'enjoying', 'a', 'turn', 'from', 'the', 'dirt', 'course', '.', '<eos>']
```

![test set attn](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/13_AIAYN_Recap/attns/attn_3.png?raw=true)


---

<p align="center">
<iframe src="https://giphy.com/embed/tBxyh2hbwMiqc" width="344" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/funny-cat-gato-gatos-tBxyh2hbwMiqc"></a></p>
</p>

---

<p align="center">
satyajit 🐱<br>:wq
</p>
