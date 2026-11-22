class BasicTokenizer:
    def train(self, text, vocabb_size, verbose=False): ...

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids): ...


tokenizer = BasicTokenizer()
print(tokenizer.encode("hii@ ô ư ă, ă"))
