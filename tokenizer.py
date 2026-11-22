class BasicTokenizer:
    def train(self, text, vocabb_size, verbose=False): ...

    def encode(self, text):
        return text.encode("utf-8")

    def decode(self, ids):
        return ids.decode("utf-8")


tokenizer = BasicTokenizer()
print(tokenizer.decode(tokenizer.encode("hii@ ô ư ă, ă")))
