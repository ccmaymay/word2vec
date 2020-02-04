This repository contains the [word2vec C code](https://github.com/tmikolov/word2vec), but with comments.  Run

```bash
    git diff original
```

to see the complete set of changes to the original code.

Cross-reference this code with the original papers:
* [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

and the essential follow-up paper:
* [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://transacl.org/ojs/index.php/tacl/article/view/570)

The following [gensim](https://radimrehurek.com/gensim/models/word2vec.html) blog posts by Radim Řehůřek are very interesting and informative:
* [Making sense of word2vec](https://rare-technologies.com/making-sense-of-word2vec/)
* [Deep learning with word2vec and gensim](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/)
* [Word2vec in Python, Part Two: Optimizing](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/)
* [Parallelizing word2vec in Python](https://rare-technologies.com/parallelizing-word2vec-in-python/)

And---shameless plug---I wrote a C++ implementation of word2vec (the skip-gram with negative sampling (SGNS) algorithm) that also supports streaming (vocabulary and embedding model are learned in one pass, see [the write-up on arXiv](https://arxiv.org/abs/1704.07463) for details):
* [athena](https://github.com/cjmay/athena) ([naive-lm-train-raw](https://github.com/cjmay/athena/blob/master/athena/naive-lm-train-raw.cpp) is the entry point for basic SGNS training)

The original content of README.txt follows the break.

---

Tools for computing distributed representtion of words
------------------------------------------------------

We provide an implementation of the Continuous Bag-of-Words (CBOW) and the Skip-gram model (SG), as well as several demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous
Bag-of-Words or the Skip-Gram neural network architectures. The user should to specify the following:
 - desired vector dimensionality
 - the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model
 - training algorithm: hierarchical softmax and / or negative sampling
 - threshold for downsampling the frequent words 
 - number of threads to use
 - the format of the output word vector file (text or binary)

Usually, the other hyper-parameters such as the learning rate do not need to be tuned for different training sets. 

The script demo-word.sh downloads a small (100MB) text corpus from the web, and trains a small word vector model. After the training
is finished, the user can interactively explore the similarity of the words.

More information about the scripts is provided at https://code.google.com/p/word2vec/
