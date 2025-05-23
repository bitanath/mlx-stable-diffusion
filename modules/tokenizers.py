import regex
import mlx.core as mx
from typing import List, Union, Dict, Tuple

class CLIPTokenizer:
    def __init__(self, vocab: Dict[str, int], merges: List[str], max_length: int = 77):
        self.max_length = max_length
        self.vocab = vocab
        
        self.bpe_ranks = {}
        start_idx = 1 if merges and merges[0].startswith('#version') else 0
        
        for i, merge in enumerate(merges[start_idx:]):
                pair = tuple(merge.split())
                self.bpe_ranks[pair] = i
        
        self.pat = regex.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex.IGNORECASE)
        self._cache = {self.bos: self.bos, self.eos: self.eos}

    @property
    def bos(self) -> str:
        return "<|startoftext|>"

    @property
    def bos_token(self) -> int:
        return self.vocab[self.bos]

    @property
    def eos(self) -> str:
        return "<|endoftext|>"

    @property
    def eos_token(self) -> int:
        return self.vocab[self.eos]

    def bpe(self, text: str) -> List[str]:
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:]))

        if not unique_bigrams:
            return unigrams

        while unique_bigrams:
            bigram = min(unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:]):
                if skip:
                    skip = False
                    continue

                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True
                else:
                    new_unigrams.append(a)

            if not skip:
                new_unigrams.append(b)

            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:]))

        self._cache[text] = unigrams
        return unigrams

    def tokenize(self, text: Union[str, List[str]], prepend_bos: bool = True, append_eos: bool = True) -> Union[List[int], List[List[int]]]:
        if isinstance(text, list):
            return [self.tokenize(t, prepend_bos, append_eos) for t in text]

        clean_text = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean_text)
        bpe_tokens = [ti for t in tokens for ti in self.bpe(t)]
        tokens = [self.vocab[t] for t in bpe_tokens]
        if prepend_bos:
            tokens = [self.bos_token] + tokens
        if append_eos:
            tokens.append(self.eos_token)

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
            if append_eos:
                tokens[-1] = self.eos_token

        return tokens

    def encode(self, text: Union[str, List[str]], padding: bool = True) -> Dict[str, mx.array]:
        if not isinstance(text, list):
            return self.encode([text], padding)

        tokens = self.tokenize(text)
        
        if padding:
            length =  self.max_length
            attention_mask = []
            
            for i, t in enumerate(tokens):
                mask = [1] * len(t)
                if len(t) < length:
                    tokens[i] = t + [self.eos_token] * (length - len(t))
                    mask = mask + [0] * (length - len(mask))
                elif len(t) > length:
                    tokens[i] = t[:length]
                    mask = mask[:length]
                attention_mask.append(mask)
        else:
            attention_mask = [[1] * len(t) for t in tokens]

        return {
            "input_ids": mx.array(tokens),
            "attention_mask": mx.array(attention_mask)
        }