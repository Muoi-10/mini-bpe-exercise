class BasicTokenizer:
    def train(self, text, vocabb_size, verbose=False): ...

    def encode(self, text): ...

    def decode(self, ids): ...


def count_pair(text):
    """text in bytes"""
    count = dict()
    for a, b in zip(text, text[1:]):
        count[(a, b)] = count.get((a, b), 0) + 1

    return sorted(count.items(), key=lambda x: x[1], reverse=True)


tokenizer = BasicTokenizer()
print(tokenizer.decode(tokenizer.encode("hii@ ô ư ă, ă")))
