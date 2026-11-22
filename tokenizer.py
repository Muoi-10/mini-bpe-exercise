class BasicTokenizer:
    def train(self, text, vocabb_size, verbose=False): ...

    def encode(self, text): ...

    def decode(self, ids): ...


tokenizer = BasicTokenizer()
print(tokenizer.decode(tokenizer.encode("hii@ ô ư ă, ă")))
