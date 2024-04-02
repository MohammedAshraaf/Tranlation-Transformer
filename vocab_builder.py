from torchtext.vocab import build_vocab_from_iterator


class VocabBuilder:
    def __init__(self, special_tokens, token_transform):
        self.token_transform = token_transform
        self.special_tokens = special_tokens

    def __yield_tokens(self, sentences, language):
        for sentence in sentences:
            yield self.token_transform[language](sentence)

    def build_vocab(self, df, language, min_freq=1):
        """
        Builds the vocab dictionary
        :param df: Pandas Dataframe that contains a column with the name equal to the given langauge.
                    that contains the text
        :param language: The language to build the vocab from. It should match a column in the given df.
        :param min_freq: Min number of occurrence for a word to be considered in the vocab

        :return: Vocab transform
        """
        vocab_transform = build_vocab_from_iterator(
            self.__yield_tokens(df[language].values, language),
            min_freq=min_freq,
            specials=self.special_tokens.keys(),
            special_first=True
        )
        vocab_transform.set_default_index(self.special_tokens['<unk>'])
        return vocab_transform
