# 09 NLP Evaluation

## Assignment

Pick any of your past code and:

1.  Implement the following metrics (either on separate models or same, your choice):
    1.  Recall, Precision, and F1 Score
    2.  BLEU
    3.  Perplexity (explain whether you are using bigram, trigram, or something else, what does your PPL score represent?)
    4.  BERTScore (here are  [1 (Links to an external site.)](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q)  [2 (Links to an external site.)](https://huggingface.co/metrics/bertscore)  examples)
2.  Once done, proceed to answer questions in the Assignment-Submission Page.
    
    Questions asked are:
    
    1.  Share the link to the readme file where you have explained all 4 metrics.
    2.  Share the link(s) where we can find the code and training logs for all of your 4 metrics
    3.  Share the last 2-3 epochs/stage logs for all of your 4 metrics separately (A, B, C, D) and describe your understanding about the numbers you're seeing, are they good/bad? Why?

## Solution

[`ttc_nlp`](https://github.com/extensive-nlp/ttc_nlp): This package was developed to keep models and datasets in an organized way. On every colab run this package is installed. It also makes sure of the package versions so there should be no breaking changes from now on.

### Text Classification Model and Evaluation

<div align="center">
<a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/ClassificationEvaluation.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/ClassificationEvaluation.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</div>
<br/>

Dataset: [SST](https://nlp.stanford.edu/sentiment/index.html)

| Model | Precision | Recall | F1 |
|--|--|--|--|
| LSTM | 0.414 | 0.357 | 0.412

		
```
Test Epoch 7/9: F1 Score: 0.41271, Precision: 0.41481, Recall: 0.35720

Classification Report
               precision    recall  f1-score   support

very negative       0.31      0.10      0.15       270
     negative       0.44      0.65      0.52       603
      neutral       0.29      0.19      0.23       376
     positive       0.38      0.57      0.46       491
very positive       0.66      0.27      0.39       385

     accuracy                           0.41      2125
    macro avg       0.41      0.36      0.35      2125
 weighted avg       0.42      0.41      0.38      2125
```

Confusion Matrix

![confusion matrix](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/sst_cm.png?raw=true)

```
Test Epoch 9/9: F1 Score: 0.38965, Precision: 0.38578, Recall: 0.37095

Classification Report
               precision    recall  f1-score   support

very negative       0.35      0.27      0.30       270
     negative       0.44      0.42      0.43       603
      neutral       0.23      0.19      0.21       376
     positive       0.36      0.52      0.43       491
very positive       0.55      0.45      0.50       385

     accuracy                           0.39      2125
    macro avg       0.39      0.37      0.37      2125
 weighted avg       0.39      0.39      0.39      2125
````

![confusion matrix of 10th epoch](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/sst_cm_10.png?raw=true)

I've taken two example to show something, Epoch 7 with F1 Score of 0.41 and Epoch 9 with F1 Score of 0.389.

So it seems like the F1 Score is decreasing, so the model is not learning, but something i did observe is that, classes Negative and Positive are more compared to the others. So even though F1 Score has decreased, but the "weighted" F1 score has increased from 0.38 to 0.39, as the model has started to focus on other classes as well, like "very negative" had only 27 correct classifications, but on 9th epoch it went to 73.

A Score of 0.41 is not that good, considering that people have gone upto 0.60+, but then those models are using [Transformers, BiDirectional LSTM with CNN](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained). But our model is simple, just by using Augmentations we have achieved a pretty good accuracy I would say.

**Stat Scores**

To Compute Precision, Recall or F1 we basically need the True Positives, True Negatives, False Positives and False Negatives, the below functions from [torchmetrics](https://github.com/PyTorchLightning/metrics) does that

```python
def _stat_scores(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate the number of tp, fp, tn, fn.
    Args:
        preds:
            An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target:
            An ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
        reduce:
            One of ``'micro'``, ``'macro'``, ``'samples'``
    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depnds on the shape of the inputs
        and the ``reduce`` parameter:
        If inputs are of the shape ``(N, C)``, then
        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce'samples'``, the returned tensors are ``(N,)`` tensors
        If inputs are of the shape ``(N, C, X)``, then
        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors
    """
    dim: Union[int, List[int]] = 1  # for "samples"
    if reduce == "micro":
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == "macro":
        dim = 0 if preds.ndim == 2 else 2

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)

    return tp.long(), fp.long(), tn.long(), fn.long()
```

**Precision**

$\text{Precision} = \frac{TP}{TP+FP}$

**Recall**

$\text{Recall} = \frac{TP}{TP+FN}$

**F1 Score**

Harmonic Mean of Precision and Recall

$\text{F1} = \frac{2\times Precison\times Recall}{Precision+Recall}$

**Intuition behind Precision and Recall**

You can think of precision as the proportion of times that when you predict its positive it actually turns out to be positive. Where as recall can be thought of as accuracy over just the positives – it’s the proportion of times you labeled positive correctly over the amount of times it was actually positive.

In the multi-label case, precision and recall are usually applied on a per category basis. That is, if you are trying to guess whether a picture has a cat or dog or other animals, you would get precision and recall for your cats and dogs separately. Then it’s just the binary case again – if you want the precision for cats, you take the number of times you guessed correctly that it was cat / the total number of times that you guessed anything was a cat. Similarly, if you want to get recall for cats, you take the number of times you guessed correctly it was a cat over the total number of times it was actually a cat.

I like to think it this way: Precision is about how precise i am, right ? like how fine i can be, so i need to make correct predictions of the class from what all i predicted it to be that class. So lets say say i predicted 100 images as cats (there can be images of dogs that i can predict as cat), and out of those 50 images were actually cats, then i have a precision of 0.5.
Recall is "just out of cat images", how many i have got right! here i cannot include dog images !


### Language Translation Model and Evaluation

<div align="center">
<a href="https://nbviewer.jupyter.org/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/TranslationTransformer.ipynb"><img alt="Open In NBViewer" src="https://img.shields.io/badge/render-nbviewer-orange?logo=Jupyter" ></a> | <a href="https://githubtocolab.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/TranslationTransformer.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</div>
<br/>

Dataset: [Multi30k](https://github.com/multi30k/dataset)

| Model | PPL | BLEU Score | BERT Score |
|--|--|--|--|
| Seq2Seq w/ Multi Head Transformer | 7.572 | 32.758 | P=0.942 R=0.939 F1=0.940 |

![blue_bert](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/bleu_bert.png?raw=true)

![crossentropy_ppl](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/cross_entropy_ppl.png?raw=true)


**Perplexity**

Perplexity comes from Information Theory, is a measurement of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample. [WikiPedia]

$PPL(p)=e^{-\sum_x{p(x)log_{e}{p(x)}}}$

But you can observe that the exponent is Cross Entropy, Hence

$\text{Cross Entropy}=p(x)log_{e}{p(x)}$
$\text{PPL}=e^{CE}$

Intuitively, perplexity can be understood as a measure of uncertainty. The perplexity of a language model can be seen as the level of perplexity when predicting the following symbol. Consider a language model with an entropy of three bits, in which each bit encodes two possible outcomes of equal probability. This means that when predicting the next symbol, that language model has to choose among $2^3=8$ possible options. Thus, we can argue that this language model has a perplexity of $8$. [Source](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)

The PPL calculated for this model was in Unigram, which was $7.572$, this would be interpreted as the model has to choose among $~8$ possible options of words to predict the next outcome. Its Good Enough ? `¯\_(ツ)_/¯`

**BLEU Score**

There's this nice interpretation of BLEU Score from [Google Cloud](https://cloud.google.com/translate/automl/docs/evaluate)

| BLEU Score | Interpretation |
|--|--|
| < 10 | Almost useless
| 10 - 19 | Hard to get the gist
| 20 - 29 | The gist is clear, but has significant grammatical errors
| 30 - 40 | Understandable to good translations
| 40 - 50 | High quality translations
| 50 - 60 | Very high quality, adequate, and fluent translations
| > 60 | Quality often better than human

BLEU first makes n-grams (basically combine n words) from the predicted sentences and compare it with the n-grams of the actual target sentences. This matching is independent of the position of the n-gram. More the number of matches, more better the model is at translating.

We got a BLEU Score of `32.758`, so it comes under "Understandable to good translation", and it is ! Note that this score was got from using unigram, bigram and trigram of the corpuses.

```python
translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu .")
>>>   
A group of people stand in front of an outdoor airport .
```

On Google Translate this gives
```
A group of people stands in front of an igloo
```

So the model got everything other than the igloo, quite possibly because it would have not encountered this meaning before.

Implementation:

```python
def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

    Arguments:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(references_corpus),\
        'The length of candidate and reference corpus should be the same'

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:  # TODO: no need to loop through the whole counter
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()
```

**BERT Score**

![bertscore architecture](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/Architecture_BERTScore.png?raw=true)

BertScore basically addresses two common pitfalls in n-gram-based metrics. Firstly, the n-gram models fail to robustly match paraphrases which leads to performance underestimation when semantically-correct phrases are penalized because of their difference from the surface form of the reference.

Each token in $x$ is matched to the most similar token in $\hat{x}$ and vice-versa for calculating Recall and Precision respectively. The matching is greedy and isolated. Precision and Recall are combined for calculating the F1 score.

The Scores we get are relative to BERT model performing on the dataset. We get a score of `0.94` pretty good ? too good to be true ? yes could be, but the validation dataset has only 1K samples.

The Model used to evaluate was `RoBERT`
```
roberta-large_L17_no-idf_version=0.3.9(hug_trans=4.8.2) P: 0.940923 R: 0.940774 F1: 0.940776
```

And here's a sample run which shows the similarity matrix generated by BERTScore

![similarity matrix](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/09_NLP_Evaluation/similarity_matrix.png?raw=true)


The BERT Score implementation was taken from [`bert_score`](https://github.com/Tiiiger/bert_score), the source code of the scoring function can be found [here](https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py)


```python
def score(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
):
    """
    BERTScore metric.
    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have 
                  multiple references, the returned score of this candidate is 
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv")
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
            else:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}", file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

    if return_hash:
        return tuple(
            [
                out,
                get_hash(model_type, num_layers, idf, rescale_with_baseline, use_custom_baseline=use_custom_baseline,),
            ]
        )

    return out
```

---
<p align="center">
<iframe src="https://giphy.com/embed/3nbxypT20Ulmo" width="480" height="355" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/coffee-morning-3nbxypT20Ulmo"></a></p>
</p>

<p align="center"><a href="https://open.spotify.com/track/1lIYP8fDGGnp91OMTUnwjV">🎶 Waqt Ki Baatein</a></p>

---

<p align="center">
:wq satyajit
</p>
