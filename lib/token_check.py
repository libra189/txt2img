import pprint
from transformers import CLIPTokenizer

class TokenCheck:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14") -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.max_len = self.tokenizer.model_max_length

    def check(self, prompt: str) -> None:
        token_len = self.len(prompt)
        result = "OK" if token_len <= self.max_len else "NG"
        tokens = self.tokens(prompt)

        print(f"{result}: token size is {token_len}(Max size: {self.max_len}).\nTokens: ", end="")
        pprint.pprint(tokens)

    def len(self, prompt: str) -> int:
        return len(self.tokenizer.tokenize(prompt))

    def tokens(self, prompt: str) -> list[str]:
        return self.tokenizer.tokenize(prompt)[0:self.max_len]
