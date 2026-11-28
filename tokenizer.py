import regex as re
import tiktoken


class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        self.merges = dict()
        self.vocab = [bytes([i]) for i in range(256)]
        merge_num = vocab_size - 256
        byte_txt = list(text.encode("utf-8"))
        for i in range(merge_num):
            top_pair = count_pair(byte_txt)[0][0]
            self.vocab.append(self.vocab[top_pair[0]] + self.vocab[top_pair[1]])
            self.merges[top_pair] = i + 256
            byte_txt = self.merge(byte_txt, top_pair)

        self.rev_merges = {v: k for k, v in self.merges.items()}
        if verbose:
            self.visual_mer()

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
        for mer in self.merges:
            new_txt = []
            skip_next = False
            for pair in zip(text, text[1:]):
                if skip_next:
                    skip_next = False
                    continue
                if pair == mer:
                    new_txt.append(self.merges[pair])
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

    def visual_mer(self):
        for pair, mer in self.merges.items():
            print(
                self.vocab[pair[0]],
                "+",
                self.vocab[pair[1]],
                "=",
                self.vocab[mer],
            )


def count_pair(text):
    """text in bytes"""
    count = dict()
    for a, b in zip(text, text[1:]):
        count[(a, b)] = count.get((a, b), 0) + 1

    return sorted(count.items(), key=lambda x: x[1], reverse=True)


class RegexTokenizer:
    def train(self, text, vocab_size, verbose=False):
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        pattern = re.compile(GPT4_SPLIT_PATTERN)

        self.merges = dict()
        self.vocab = [bytes([i]) for i in range(256)]
        merge_num = vocab_size - 256
        byte_txt = [t.encode("utf-8") for t in re.findall(pattern, text)]
        for i in range(merge_num):
            top_pair = self.count_pair_lst(byte_txt)[0][0]
            self.vocab.append(self.vocab[top_pair[0]] + self.vocab[top_pair[1]])
            self.merges[top_pair] = i + 256
            byte_txt = self.merge_lst(byte_txt, top_pair)

        self.rev_merges = {v: k for k, v in self.merges.items()}

        if verbose:
            self.visualize_merges()

    def count_pair_lst(self, byte_txt_lst):
        count = dict()
        for text in byte_txt_lst:
            for a, b in zip(text, text[1:]):
                count[(a, b)] = count.get((a, b), 0) + 1
        return sorted(count.items(), key=lambda x: x[1], reverse=True)

    def merge_lst(self, byte_txt_lst, pair):
        new_lst = []
        for text in byte_txt_lst:
            new_txt = []
            text = list(text)
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
            new_lst.append(new_txt)

        return new_lst

    def encode(self, text):
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        pattern = re.compile(GPT4_SPLIT_PATTERN)
        byte_txt_lst = [t.encode("utf-8") for t in re.findall(pattern, text)]
        for mer in self.merges:
            new_lst = []
            for text in byte_txt_lst:
                text = list(text)
                new_txt = []
                skip_next = False
                for pair in zip(text, text[1:]):
                    if skip_next:
                        skip_next = False
                        continue
                    if pair == mer:
                        new_txt.append(self.merges[pair])
                        skip_next = True
                    else:
                        new_txt.append(pair[0])
                if not skip_next:
                    new_txt.append(text[-1])
                new_lst.append(new_txt)
            byte_txt_lst = new_lst.copy()

        return [num for row in byte_txt_lst for num in row]

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

    def visualize_merges(self):
        for pair, mer in self.merges.items():
            print(
                self.vocab[pair[0]],
                "+",
                self.vocab[pair[1]],
                "=",
                self.vocab[mer],
            )


class GPT4_tokenizer(RegexTokenizer):
    def __init__(self):
        self.merges = self.recover_merges(enc._mergeable_ranks)
        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.inverse_merges = {v: k for k, v in self.merges.items()}

    def encode(self, text):
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        pattern = re.compile(GPT4_SPLIT_PATTERN)
        byte_txt_lst = [
            self.convert_to_gpt(t.encode("utf-8")) for t in re.findall(pattern, text)
        ]

        for mer in self.merges:
            new_lst = []
            for text in byte_txt_lst:
                text = list(text)
                new_txt = []
                skip_next = False
                for pair in zip(text, text[1:]):
                    if skip_next:
                        skip_next = False
                        continue
                    if pair == mer:
                        new_txt.append(self.merges[pair])
                        skip_next = True
                    else:
                        new_txt.append(pair[0])
                if not skip_next:
                    new_txt.append(text[-1])
                new_lst.append(new_txt)
            byte_txt_lst = new_lst.copy()

        return [num for row in byte_txt_lst for num in row]

    def decode(self, txt_lst): ...

    def convert_to_gpt(self, byte_txt):
        new = [self.byte_shuffle[c] for c in list(byte_txt)]
        return new

    def decode(self, ids):
        change = True
        while change:
            new_ids = []
            change = False
            for id in ids:
                if self.inverse_merges.get(id):
                    new_ids += list(self.inverse_merges[id])
                    change = True
                else:
                    new_ids.append(id)

            ids = new_ids.copy()
        ids = self.convert_to_utf(ids)
        print(ids)
        text = bytes(ids).decode("utf-8")

        return text

    def convert_to_utf(self, ids):
        new = [self.inverse_byte_shuffle[c] for c in list(ids)]
        return new

    def bpe(self, mergeable_ranks, token, max_rank):
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            parts = (
                parts[:min_idx]
                + [parts[min_idx] + parts[min_idx + 1]]
                + parts[min_idx + 2 :]
            )
        return parts

    def recover_merges(self, mergeable_ranks):
        merges = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue
            pair = tuple(self.bpe(mergeable_ranks, token, max_rank=rank))
            assert len(pair) == 2
            merges[(mergeable_ranks[pair[0]], mergeable_ranks[pair[1]])] = rank
        return merges


with open("taylorswift.txt", "r", encoding="utf-8") as f:
    content = f.read()

# tokenizer = BasicTokenizer()
# tokenizer.train(content, 270, verbose=True)

# regex_tokenizer = RegexTokenizer()
# regex_tokenizer.train(content, 300, verbose=True)

enc = tiktoken.get_encoding("cl100k_base")  # this is the GPT-4 tokenizere
gpt_tokenizer = GPT4_tokenizer()
byte_txt = "hello".encode("utf-8")
en = gpt_tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")
print(gpt_tokenizer.decode(en))
