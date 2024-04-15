
import re


class TextPromptFilter:
    def __init__(self, remove_nl=False) -> None:
        self.remove_nl = remove_nl
        self._1time_remove_nl = False

    def __ror__(self, txt):

        txt = re.sub(r"\n?([\t ]*#[^\n]+)", "", txt, re.DOTALL | re.M)  # remove comment
        if self.remove_nl or self._1time_remove_nl:
            txt = " ".join([item.strip() for item in txt.splitlines()])

        self._1time_remove_nl = False
        return txt.strip()

    def __call__(self, remove_nl):
        self._1time_remove_nl = remove_nl
        return self


