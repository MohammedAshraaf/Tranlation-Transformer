import re
from sklearn.model_selection import train_test_split


class Preprocessing:
    @staticmethod
    def clean_text(text):
        """
        Removes unwanted characters, symbols, and extra spaces
        :param text: a string representing a sentence.

        :return: The cleaned text
        """
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    @staticmethod
    def preprocess_df(df, languages):
        """
        Applies preprocessing steps on the data
        :param df: Pandas DataFrame that contains the data. It should have a column for each passed language
                    that contains the text.
        :param languages: list of languages where their names matches a column in the given df.

        :return: the df after applying the preprocessing
        """
        for language in languages:
            df[language] = df[language].astype(str)
            df[language] = df[language].apply(Preprocessing.clean_text)
            df[f'{language}_len'] = df[language].apply(lambda x: len(x.split()))
        return df

    @staticmethod
    def filter_data_by_length(df, language, min_length, max_length):
        """
        Filters the df based on the length of the sentences of the given language
        :param df: Pandas DataFrame that contains the data. It should have a column named {language}_len,
                    which has the length of the current row's text for the given language.
        :param language: The language to be used for filtering the text based on.
                            it should match a column in the df called {language}_len
        :param min_length: Minimum number of words per text
        :param max_length: Maximum number of words per text

        :return: The filtered df
        """
        df = df[(df[f'{language}_len'] >= min_length) & (df[f'{language}_len'] <= max_length)]
        return df

    @staticmethod
    def split_df(df, test_size, seed):
        """
        Splits the data into training, validation, and testing
        :param df: Pandas DataFrame that contains the data
        :param test_size: the size of the test set which would be equally split between validation and testing
        :param seed: Seed to be used when splitting for replicating the experiment

        :return: The three splits, training, validation, and testing
        """
        # Split data into training and testing sets (70% train, 30% test)
        train, test = train_test_split(df, test_size=test_size, random_state=seed)

        # Further split training data into training and validation sets (70% of train data for training, 30% for validation)
        val, test = train_test_split(test, test_size=0.5, random_state=seed)

        return train, val, test
