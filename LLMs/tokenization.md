# Tokenization in LLMs

## Question
What is tokenization and why is it important for LLMs? Compare different tokenization approaches.

## Answer

### Overview
Tokenization converts text into discrete units (tokens) that can be processed by language models.

**Raw text → Tokens → Token IDs → Embeddings → Model**

### Why Tokenization Matters

**Tradeoff:**
- Character-level: Small vocab, long sequences
- Word-level: Large vocab, short sequences
- Subword: **Optimal balance**

## Subword Tokenization Algorithms

## 1. Byte Pair Encoding (BPE)

**Used by:** GPT-2, GPT-3, RoBERTa, CodeX

### Algorithm

1. Start with character vocabulary
2. Find most frequent pair
3. Merge into new token
4. Repeat

**Example:**
```
"low low low" → frequent pair ('l','o')
→ merge to 'lo'
→ "lo w lo w lo w"
→ frequent pair ('lo','w')
→ merge to 'low'
```

### PyTorch Usage

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
# ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
# 'Ġ' represents space

token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)
```

## 2. WordPiece

**Used by:** BERT, DistilBERT

**Difference from BPE:** Merges based on **likelihood increase**, not frequency.

### Special Tokens

```
[CLS]: Start of sequence
[SEP]: Separator
[PAD]: Padding
[UNK]: Unknown
[MASK]: Masked (for MLM)
```

### Usage

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "unbelievable happiness"
tokens = tokenizer.tokenize(text)
# ['un', '##believable', 'happiness']
# '##' indicates continuation
```

## 3. SentencePiece

**Used by:** T5, ALBERT, Llama, Mistral

**Key Innovation:** Language-agnostic, treats input as raw byte stream.

**Whitespace handling:**
```
"Hello world" → ['▁Hello', '▁world']
```
`▁` represents space, making tokenization **reversible**.

### Usage

```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')

text = "translate English to German: Hello"
tokens = tokenizer.tokenize(text)
# ['▁translate', '▁English', '▁to', '▁German', ':', '▁Hello']
```

## Comparison

| Method | Used By | Special Chars | Reversible |
|--------|---------|---------------|------------|
| **BPE** | GPT-2/3 | `Ġ` (space) | Yes |
| **WordPiece** | BERT | `##` (continuation) | Yes |
| **SentencePiece** | T5, Llama | `▁` (space) | **Yes** |

## Vocabulary Sizes

| Model | Tokenizer | Vocab Size |
|-------|-----------|------------|
| **BERT** | WordPiece | 30,522 |
| **GPT-2** | BPE | 50,257 |
| **T5** | SentencePiece | 32,000 |
| **Llama 2** | SentencePiece | 32,000 |
| **GPT-4** | BPE | ~100,000 |

## Byte-Level BPE

**Used by:** GPT-2, GPT-3

**Innovation:** Operates on bytes, not unicode
- Can represent **any** text (any language, code, etc.)
- Vocabulary: 256 base + merges

## Practical Considerations

### 1. Special Tokens

```python
special_tokens = {
    'pad_token': '[PAD]',
    'unk_token': '[UNK]',
    'bos_token': '<s>',
    'eos_token': '</s>',
}
```

### 2. Sequence Length

Token count ≠ word count:
```python
text = "unbelievable"
tokens = ['un', '##believable']  # 2 tokens
```

**Max sequence lengths:**
- BERT: 512 tokens
- GPT-2: 1024
- GPT-3: 2048
- GPT-4: 8k-32k

### 3. Multilingual

For multilingual models:
- Use SentencePiece
- Increase vocab size (50k-100k)

## Common Pitfalls

❌ **Not adding special tokens**
```python
# Wrong
tokens = tokenizer.tokenize(text)
```

✅ **Correct**
```python
tokens = tokenizer.encode(text, add_special_tokens=True)
```

❌ **Inefficient looping**
```python
# Slow
for text in texts:
    tokens = tokenizer.encode(text)
```

✅ **Batch encoding**
```python
# Fast
tokens = tokenizer(texts, padding=True, return_tensors='pt')
```

## Key Takeaways

1. **Subword tokenization is standard** (BPE, WordPiece, SentencePiece)
2. **No OOV issues** with subword methods
3. **BPE**: GPT family
4. **WordPiece**: BERT family
5. **SentencePiece**: T5, Llama (language-agnostic)
6. **Vocab size:** 30k-50k typical
7. **Always use tokenizer from pretrained model**

## Tags
#LLMs #Tokenization #BPE #WordPiece #SentencePiece #NLP

## Difficulty
Medium

## Related Questions
- How do transformers process text?
- Explain embedding layers in LLMs
