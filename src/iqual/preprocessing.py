import re
import unicodedata
import numpy as np
from nltk.stem import WordNetLemmatizer



def remove_accents(text):
    """"""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if (unicodedata.combining(c) == 0)])


def remove_extra_whitespaces(s):
    """
    Removes extra spaces from strings
    """
    return re.sub(" +", " ", s).strip()


def remove_extra_newlines(s):
    """
    Removes extra newlines from strings
    """
    return re.sub("\n+", "\n", s)


def remove_extra_tabs(s):
    """
    Removes extra tabs from strings
    """
    return re.sub("\t+", "\t", s)


def remove_extra_spaces(s):
    """
    Removes extra spaces from strings
    """
    s = remove_extra_whitespaces(s)
    s = remove_extra_newlines(s)
    s = remove_extra_tabs(s)
    return s


def split_after_colon(text, max_left=10):
    """
    Splits text after every colon.
    """
    splits = re.split("ঃ|:", text)
    if len(splits) == 2 and len(splits[0]) <= max_left:
        text = splits[-1]
    return text.strip()


def lowercase(text):
    """Converts all characters to lowercase"""
    if isinstance(text, str):
        return text.lower()
    else:
        return text


def remove_quotes(string):
    """
    Removes all quotes Single(') or Double(") from strings
    """
    pattern = re.compile("'|\"")
    return re.sub(pattern, "", string)


def remove_special_chars(s):
    """
    Removes the following special characters from strings
    """
    s = re.sub("[\\\/!,*)@#%(&$_.?^-]", " ", s)
    s = remove_quotes(s)
    return re.sub(" +", " ", s).strip()


def remove_space_between_numbers(text):
    """
    Removes whitespaces between numbers
    """
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    return text


def remove_dashes(x):
    """
    Removes dashes from strings
    """
    x = re.sub("-+", " ", x)
    x = re.sub(" +", " ", x)
    return x


def remove_commas_between_numbers(text):
    """
    Removes commas between numbers
    """
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1\2", text)
    return text


def keep_alnum_only(text):
    """
    Removes all non-alphanumeric characters from strings
    """
    return "".join([a for a in text if a.isalnum()])


def keep_whitespace_alnum(text):
    """
    Removes all non-alphanumeric characters from strings
    Keeps whitespaces, and alpha-numeric characters
    """
    if isinstance(text, str):
        text = "".join([a if a.isalnum() else " " for a in text])
        text = remove_extra_spaces(text)
        return text
    else:
        return text


def count_words(text):
    """
    Counts the number of words in a string
    """
    text = keep_whitespace_alnum(text)
    return len([w for w in text.split() if w.isalnum()])


def mask_digits(s):
    """
    Experimental function to mask digits in strings
    Replaces all digits with '#'
    """
    s = "".join(["#" if x.isdigit() else x for x in s])
    digit_counts = list(
        np.unique([(i.end() - i.start()) for i in list(re.finditer("#+", s))])
    )
    digit_counts.sort()
    digit_counts.reverse()
    for d in digit_counts:
        s = re.sub("".join(["#" for _ in range(d)]), f" {d}# ", s)
    return re.sub(" +", " ", s).strip()


def remove_stopwords(text: str, stopwords):
    """
    Removes all stopwords from strings
    """
    return " ".join([t for t in text.split() if t not in stopwords])


def remove_short_words(text: str, min_length=2):
    """
    Removes all words with length less than min_length
    """
    return " ".join([t for t in text.split() if len(t) >= min_length])

def lemmatize(text: str):
    """
    Lemmatizes the text
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    return " ".join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])