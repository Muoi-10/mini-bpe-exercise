class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        self.merges = dict()
        vocab = [i for i in range(256)]
        merge_num = vocab_size - 256
        byte_txt = list(text.encode("utf-8"))
        for i in range(merge_num):
            top_pair = count_pair(byte_txt)[0][0]
            vocab.append(top_pair)
            self.merges[top_pair] = i + 256
            byte_txt = self.merge(byte_txt, top_pair)

        self.rev_merges = {v: k for k, v in self.merges.items()}

    def merge(self, text, pair):
        new_txt = []
        skip_next = False
        for i in range(len(text) - 1):
            if skip_next:
                skip_next = False
                continue
            if text[i : i + 2] == list(pair):
                new_txt.append(self.merges[pair])
                skip_next = True
            else:
                new_txt.append(text[i])
        if not skip_next:
            new_txt.append(text[-1])
        return new_txt

    def encode(self, text):
        text = list(text.encode("utf-8"))
        change = True
        while change:
            new_txt = []
            change = False
            skip_next = False
            for pair in zip(text, text[1:]):
                if skip_next:
                    skip_next = False
                    continue
                if self.merges.get(pair):
                    new_txt.append(self.merges[pair])
                    change = True
                    skip_next = True
                else:
                    new_txt.append(pair[0])
            if not skip_next:
                new_txt.append(text[-1])
            text = new_txt.copy()

        return text

    def decode(self, ids):
        change = True
        while change:
            new_ids = []
            change = False
            for id in ids:
                if self.rev_merges.get(id):
                    new_ids += list(self.rev_merges[id])
                    change = True
                else:
                    new_ids.append(id)

            ids = new_ids.copy()
        text = bytes(ids).decode("utf-8")

        return text


def count_pair(text):
    """text in bytes"""
    count = dict()
    for a, b in zip(text, text[1:]):
        count[(a, b)] = count.get((a, b), 0) + 1

    return sorted(count.items(), key=lambda x: x[1], reverse=True)


tokenizer = BasicTokenizer()
tokenizer.train("ă ơ ơ hsdf????", 260)
print(tokenizer.merges)
a = tokenizer.encode("?? ơ ơ")
print(tokenizer.decode(a))
