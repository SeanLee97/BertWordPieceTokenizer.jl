# BertWordPieceTokenizer

Load BERT WordPiece Tokenizer


## Installation


```julia
julia> using Pkg; Pkg.add("BertWordPieceTokenizer")
```


## Usage

### Initialize

```julia
import BertWordPieceTokenizer as BWP

# initialize tokenizer from local file
BWP.init("/path/to/vocab.txt")
BWP.init("/path/to/vocab.txt", do_lowercase=true)
BWP.init("/path/to/vocab.txt", do_lowercase=false)

# initialize tokenizer from HTTP URL
BWP.init("https://huggingface.co/bert-base-uncased/raw/main/vocab.txt", do_lowercase=true)

# initialize tokenizer from HTTP URL, and cache vocabulary to a local file
BWP.init("https://huggingface.co/bert-base-uncased/raw/main/vocab.txt", cache_path="/path/to/vocab.txt", do_lowercase=true)
```

### Tokenize

After initializing, you can tokenize text using `BWP.tokenize`, and encode text using `BWP.encode`

```julia
# tokenize text
tokens = BWP.tokenize("i like apples")

# encode text
token_ids, segment_ids = BWP.encode("i like apples")
```

### Truncation

You can specify max length and truncation strategy using `BWP.enable_truncation`

```julia
BWP.enable_truncation(512, truncation="post")
BWP.enable_truncation(512, truncation="pre")
```


## Benchmark

1) Julia

```julia
julia> using TimeIt

julia> import BertWordPieceTokenizer as BWP

julia> BWP.init("https://huggingface.co/bert-base-uncased/raw/main/vocab.txt", cache_path="bert_uncased_vocab.txt", do_lowercase=true)

julia> BWP.encode("I like apples")
([101, 1045, 2066, 18108, 102], [0, 0, 0, 0, 0])

julia> @timeit BWP.encode("I like apples")
100000 loops, best of 3: 6.77 µs per loop
```


2) Pure Python

```python
In [1]: from bert4keras.tokenizers import Tokenizer

In [2]: tokenizer = Tokenizer("bert_uncased_vocab.txt", do_lower_case=True)

In [3]: tokenizer.encode("I like apples")
Out[3]: ([101, 1045, 2066, 18108, 102], [0, 0, 0, 0, 0])

In [4]: %timeit tokenizer.encode("I like apples")
48.7 µs ± 1.93 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

3) Rust binding for Python


```python
In [1]: from tokenizers import BertWordPieceTokenizer

In [2]: tokenizer = BertWordPieceTokenizer("bert_uncased_vocab.txt", lowercase=True)

In [3]: tokenizer.encode("I like apples")
Out[3]: Encoding(num_tokens=5, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])

In [4]: %timeit tokenizer.encode("I like apples")
23.8 µs ± 274 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
