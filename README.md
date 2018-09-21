# metapy: (experimental) Python bindings for [MeTA][meta]

[![Build Status](https://travis-ci.org/meta-toolkit/metapy.svg?branch=master)](https://travis-ci.org/meta-toolkit/metapy)

[![Windows Build Status](https://ci.appveyor.com//api/projects/status/github/meta-toolkit/metapy?svg=true&branch=master)](https://ci.appveyor.com/project/skystrife/metapy)

This project provides Python (2.7 and 3.x are supported) bindings for the
MeTA toolkit. They are still very much under construction, but the goal is
to make it seamless to use MeTA's components within any Python application
(e.g., a Django or Flask web app).

This project is made possible by the excellent [pybind11][pybind11]
library.

## Getting Started (the easy way)

```bash
# Ensure your pip is up to date
pip install --upgrade pip

# install metapy!
pip install metapy
```

This should work on Linux, OS X, and Windows with pretty much any recent
Python version >= 2.7. On Linux, make sure to update your `pip` to version
8.1 so you can install from a binary package---this will save you a lot of
time.

## Getting Started (the hard way)

You will, of course, need Python installed. You will also need its headers
to be installed as well, so look for a `python-dev` or similar package for
your system. Beyond that, you'll of course need to satisfy the requirements
for [building MeTA itself][build-guide].

This repository should have everything you need to get started. You should
ensure that you've fetched all of the submodules first, though:

```bash
git submodule update --init --recursive
```

Once that's done, you should be able to build the library like so:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

You can force building against a specific version of Python if you happen
to have multiple versions installed by specifying
`-DMETAPY_PYTHON_VERSION=x.y` when invoking `cmake`.

The module should be written to `metapy.so` in the build directory.

[meta]: https://meta-toolkit.org
[pybind11]: https://github.com/pybind/pybind11
[build-guide]: https://meta-toolkit.org/setup-guide.html

## How to use?
Adding a few documentation for metapy API that has copied from CS410 programming assignment 1.

### Start
Let's start by importing metapy. Open a terminal and type
```bash
python
``` 
to get started

```python
#import the MeTA python bindings
import metapy
#If you'd like, you can tell MeTA to log to stderr so you can get progress output when running long-running function calls.
metapy.log_to_stderr()
```


Now, let's create a document with some content to experiment on
```python
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
```

### Tokenization
MeTA provides a stream-based interface for performing document tokenization.
Each stream starts off with a Tokenizer object, and in most cases you should use the Unicode standard aware ICUTokenizer.

```python
tok = metapy.analyzers.ICUTokenizer()
```

Tokenizers operate on raw text and provide an Iterable that spits out the individual text tokens.
Let's try running just the ICUTokenizer to see what it does.

```python
tok.set_content(doc.content()) # this could be any string
tokens = [token for token in tok]
print(tokens)
```

One thing that you likely immediately notice is the insertion of these pseudo-XML looking <s> and </s> tags.
These are called “sentence boundary tags”.
As a side-effect, a default-construted ICUTokenizer discovers the sentences in a document by delimiting them with the sentence boundary tags.
Let's try tokenizing a multi-sentence document to see what that looks like.

```python
doc.content("I said that I can't believe that it only costs $19.95! I could only find it for more than $30 before.")
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Most of the information retrieval techniques you have likely been learning about in this class don't need to concern themselves with finding the boundaries between separate sentences in a document, but later today we'll explore a scenario where this might matter more.
Let's pass a flag to the ICUTokenizer constructor to disable sentence boundary tags for now.

```python
tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

As mentioned earlier, MeTA treats tokenization as a streaming process, and that it starts with a tokenizer.
It is often beneficial to modify the raw underlying tokens of a document, and thus change its representation.
The “intermediate” steps in the tokenization stream are represented with objects called Filters.
Each filter consumes the content of a previous filter (or a tokenizer) and modifies the tokens coming out of the stream in some way.
Let's start by using a simple filter that can help eliminate a lot of noise that we might encounter when tokenizing web documents: a LengthFilter.

```python
tok = metapy.analyzers.LengthFilter(tok, min=2, max=30)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Here, we can see that the LengthFilter is consuming our original ICUTokenizer.
It modifies the token stream by only emitting tokens that are of a minimum length of 2 and a maximum length of 30.
This can get rid of a lot of punctuation tokens, but also excessively long tokens such as URLs.

### Stopword removal and stemming

Another common trick is to remove stopwords. In MeTA, this is done using a ListFilter.

```bash
wget -nc https://raw.githubusercontent.com/meta-toolkit/meta/master/data/lemur-stopwords.txt
```

```python
tok = metapy.analyzers.ListFilter(tok, "lemur-stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Here we've downloaded a common list of stopwords and created a ListFilter to reject any tokens that occur in that list of words.
You can see how much of a difference removing stopwords can make on the size of a document's token stream!

Another common filter that people use is called a stemmer, or lemmatizer.
This kind of filter tries to modify individual tokens in such a way that different inflected forms of a word all reduce to the same representation.
This lets you, for example, find documents about a “run” when you search “running” or “runs”.
A common stemmer is the Porter2 Stemmer, which MeTA has an implementation of.
Let's try it!

```python
tok = metapy.analyzers.Porter2Filter(tok)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

### N-grams

Finally, after you've got the token stream configured the way you'd like, it's time to analyze the document by consuming each token from its token stream and performing some actions based on these tokens.
In the simplest case, our action can simply be counting how many times these tokens occur.
For clarity, let's switch back to a simpler token stream first.
We will write a token stream that tokenizes with ICUTokenizer, and then lowercases each token.

```python
tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
tok = metapy.analyzers.LowercaseFilter(tok)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Now, let's count how often each individual token appears in the stream.
This representation is called “bag of words” representation or “unigram word counts”.
In MeTA, classes that consume a token stream and emit a document representation are called Analyzers.

```python
ana = metapy.analyzers.NGramWordAnalyzer(1, tok)
print(doc.content())
unigrams = ana.analyze(doc)
print(unigrams)
```

If you noticed the name of the analyzer, you might have realized that you can count not just individual tokens, but groups of them.
“Unigram” means “1-gram”, and we count individual tokens. “Bigram” means “2-gram”, and we count adjacent tokens together as a group.
Let's try that now.

```python
ana = metapy.analyzers.NGramWordAnalyzer(2, tok)
bigrams = ana.analyze(doc)
print(bigrams)
```

Now the individual “tokens” we're counting are pairs of tokens.
Sometimes looking at n-grams of characters is useful.

```python
tok = metapy.analyzers.CharacterTokenizer()
ana = metapy.analyzers.NGramWordAnalyzer(4, tok)
fourchar_ngrams = ana.analyze(doc)
print(fourchar_ngrams)
```

### POS tagging

Now, let's explore something a little bit different.
MeTA also has a natural language processing (NLP) component, which currently supports two major NLP tasks: part-of-speech tagging and syntactic parsing.
POS tagging is a task in NLP that involves identifying a type for each word in a sentence.
For example, POS tagging can be used to identify all of the nouns in a sentence, or all of the verbs, or adjectives, or…
This is useful as first step towards developing an understanding of the meaning of a particular sentence.
MeTA places its POS tagging component in its “sequences” library.
Let's play with some sequences first to get an idea of how they work.
We'll start of by creating a sequence.

```python
seq = metapy.sequence.Sequence()
```

Now, we can add individual words to this sequence.
Sequences consist of a list of Observations, which are essentially (word, tag) pairs.
If we don't yet know the tags for a Sequence, we can just add individual words and leave the tags unset.
Words are called “symbols” in the library terminology.

```python
for word in ["The", "dog", "ran", "across", "the", "park", "."]:
    seq.add_symbol(word)
print(seq)
```

The printed form of the sequence shows that we do not yet know the tags for each word.
Let's fill them in by using a pre-trained POS-tagger model that's distributed with MeTA.

```bash
wget -nc https://github.com/meta-toolkit/meta/releases/download/v3.0.1/greedy-perceptron-tagger.tar.gz
tar xvf greedy-perceptron-tagger.tar.gz
```

```python
tagger = metapy.sequence.PerceptronTagger("perceptron-tagger/")
tagger.tag(seq)
print(seq)
```

Each tag indicates the type of a word, and this particular tagger was trained to output the tags present in the Penn Treebank tagset.
But what if we want to POS-tag a document?

```python
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
tok = metapy.analyzers.ICUTokenizer() # keep sentence boundaries!
tok = metapy.analyzers.PennTreebankNormalizer(tok)
tok.set_content(doc.content())
tokens = [token for token in tok]
print(tokens)
```

Now, we will write a function that can take a token stream that contains sentence boundary tags and returns a list of Sequence objects.
We will not include the sentence boundary tags in the actual Sequence objects.

```python
def extract_sequences(tok):
    sequences = []
    for token in tok:
        if token == '<s>':
            sequences.append(metapy.sequence.Sequence())
        elif token != '</s>':
            sequences[-1].add_symbol(token)
    return sequences

doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
tok.set_content(doc.content())
for seq in extract_sequences(tok):
    tagger.tag(seq)
    print(seq)
```

### Config.toml file: setting up a pipeline

In practice, it is often beneficial to combine multiple feature sets together.
We can do this with a MultiAnalyzer. Let's combine unigram words, bigram POS tags, and rewrite rules for our document feature representation.
We can certainly do this programmatically, but doing so can become tedious quite quickly.
Instead, let's use MeTA's configuration file format to specify our analyzer, which we can then load in one line of code.
MeTA uses TOML configuration files for all of its configuration. If you haven't heard of TOML before, don't panic! It's a very simple, readable format.
Open a text editor to see how it looks like and copy the text below, but be careful not to modify the contents.


```
#Add this as a config.toml file to your project directory
stop-words = "lemur-stopwords.txt"

[[analyzers]]
method = "ngram-word"
ngram = 1
filter = "default-unigram-chain"

[[analyzers]]
method = "ngram-pos"
ngram = 2
filter = [{type = "icu-tokenizer"}, {type = "ptb-normalizer"}]
crf-prefix = "crf"

[[analyzers]]
method = "tree"
filter = [{type = "icu-tokenizer"}, {type = "ptb-normalizer"}]
features = ["subtree"]
tagger = "perceptron-tagger/"
parser = "parser/"
```

Each [[analyzers]] block defines another analyzer to combine for our feature representation.
Since “ngram-word” is such a common analyzer, we have defined some default filter chains that can be used with shortcuts.
“default-unigram-chain” is a filter chain suitable for unigram words; “default-chain” is a filter chain suitable for bigram words and above.
We can now load an analyzer from this configuration file:

```python
ana = metapy.analyzers.load('config.toml')
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
print(ana.analyze(doc))
```