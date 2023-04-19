import re
import yaml

class FixIndentDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(FixIndentDumper, self).increase_indent(flow, False)

class Txt2Prompt:
    prompt: list[str]
    negative_prompt: list[str]

    def __toList(self, txt: str) -> list[str]:
        words = txt.split(",")
        words = [w for w in words if len(w) > 0]
        words = [re.sub("_", " ", w) for w in words]
        words = [re.sub("-", " ", w) for w in words]
        words = [re.sub("\s{2,}", " ", w) for w in words]
        words = [w.lstrip() for w in words]
        words = [w.rstrip() for w in words]
        return words

    def load(self, prompt: str, negative_prompt: str = "") -> None:
        if (len(prompt) == 0):
            raise Exception("String length must be at >=1.")

        self.prompt = self.__toList(prompt)
        self.negative_prompt = self.__toList(negative_prompt) if len(negative_prompt) > 0 else []

    def save(self, file_path: str) -> None:
        data: dict[str, list[str]] = {}

        data["prompt"] = self.prompt
        if (len(self.negative_prompt) > 0):
            data["negative_prompt"] = self.negative_prompt

        with open(file_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False, width=1000, Dumper=FixIndentDumper))
