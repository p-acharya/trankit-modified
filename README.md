# Pāṇinian Grammar-Inspired Interlingua for Neural Machine Translation

## Machine Translation for low resource languages using an Interlingua

Panini defines morphosyntactic properties in the form of *karaka* relations and the morphological properties of the words in a sentence. *Karaka* is a form of syntactic analysis where we capture relationships between one word pair at a time in a sentence.

Neural taggers such as Udify and Trankit provide a multitask learning setup that predicts both dependency parsing and morphological tags.

There are different schemes for dependency parsing. UD is the most widely adopted as it is language agnostic. Traditionally *Karaka* tags also follow a dependency scheme which is hypothesized to be better suited for Indian languages.

### Morphosyntactic Parser

1. Formulated as a multitask parser
2.  **Input**: A sentence 
3. **Output**: 1 label per word per task
4. Possible Tasks
	- Syntactic Features
		- Dependency head prediction
		- Dependency label prediction - Ex: Karta (subject), Karma (object)
			- Ex : `rsp`, `ras-neg`, `rh`, `k2`, `lwg_psp`, `rt`, `k2g`, `modwq`, `mk1`, `k1s`, `lwgneg`, `k4`, `undef`, `k5`, `k2s`, `nmodk1inv`, `lwgrp`

	- Morphological features
		- Category - Ex: noun, avyaya, adjective, verb
		- Gender - Ex: male, prediction
		- Number - Ex: singular, plural
		- Case - Ex: oblique/direct
		- Vibhakthi
		- TAM (Tense Aspect Mood)
		- Person - Ex: 1st person, 2nd person, 3rd person
		- POS tag prediction

13. We use a combination of these features for our experiments.
14. We use a deep biaffine parser ([Dozat, 2016](https://arxiv.org/abs/1611.01734)) for syntactic features and a feed-forward classifier for morphological features


### Morphological Complexity
| Morpheme | HINDI | KANNADA |
| -------- | ----- | ------- |
| CAT      | 26    | 22      |
| POS      | 33    | 49      |
| TAM      | 35    | 4565    |
| VIB      | 887   | 4389    |
| GEN      | 7     | 6       |
| PERS     | 8     | 5       |
| CASE     | 7     | 4       |
| NUM      | 6     | 5       |
| DEPREL   | 81    | 92      |

### Metrics

- UAS: UAS is the performance score of a biaffine parser which predicts the dependency head 
- LAS: LAS is the performance score of two biaffine parsers, where one predicts the dependency head, and the other predicts the dependency relation 
- AllTags: AllTags is the average performance score measured across all the feed-forward classifiers

### Dataset
- Hindi dataset
	- Train size: 14089 sentences
	- Dev size: 1743 sentences
	- Test size: 1804 sentences
- Kannada dataset
	- Train size: 13088 sentences
	- Dev size: 2801 sentences
	- Test size: 2789 sentences

### Morphosyntactic Parsing
For this experiment, we use the default configuration of Trankit, where we have a single classifier that predicts all the morphological features.
The output of this classifier will be a string of form `case-{abc}_num-{def}_vib-{ghi}_gen-{jkl}_tam-{mno}_pers-{xyz}`. We refer to this as a composite classifier setting.

#### Composite Configuration
||
|------------------------------------------------------------------------------ |
| Dependency Head                                                                |
| Dependency Label                                                               |
| CAT                                                                            |
| POS                                                                            |
| `case_num_vib_gen_tam_pers`|

#### Loss function

$$L({\theta}) = L_{head} + L_{label} + \sum_1^n L_{c_{i}}$$
where $L({\theta})$ denotes the total loss function, $L_{head}$ denotes the loss function of the dependency head prediction, $L_{label}$ denotes the loss function of the dependency label prediction, and $L_{c_{i}}$ denotes the loss function of each morphological classifier.

In other words, the total loss is the sum of the loss functions for the syntactic features and the morphological features.
$$L({\theta}) = L_{syntactic} + L_{morphological}$$

### Adding multiple Classifiers
- Using the modified trankit code, update the `CLASSES` and the `CLASSIFIER_NAMES` arrays in `custom_classifiers.py`.
- `CLASSES` is a list of classifiers where each element of the array is a list. If you want to add a classifier that predicts `vib` and `num` together,  then add `[“vib”,”num”]` to `CLASSES` array
- `CLASS_NAMES` is an array of names of each classifier in `CLASSES` array
- You can modify `ignore_upos_xpos` variable in the same file to specify whether you want XPOS and UPOS to be considered while computing the loss
- run `pip install -e .  && python3 train.py` to train the model 
