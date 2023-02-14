import re
import unicodedata
import numpy as np
import num2words

# List all functions
__all__ = [
    "remove_accents",
    "remove_extra_whitespaces",
    "remove_extra_newlines",
    "remove_extra_tabs",
    "remove_extra_spaces",
    "split_after_colon",
    "lowercase",
    "remove_quotes",
    "remove_special_chars",
    "remove_space_between_numbers",
    "remove_dashes",
    "remove_commas_between_numbers",
    "replace_numbers",
    "keep_alnum_only",
    "keep_whitespace_alnum",
    "count_words",
    "replace_contractions",
    "mask_digits",
    "remove_stopwords",
    "remove_short_words",
    "replace_whole_words"
]

def remove_accents(text):
    """"""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nfkd_form if (unicodedata.combining(c)==0)])

def remove_extra_whitespaces(s):
    """
    Removes extra spaces from strings
    """
    return re.sub(" +"," ",s).strip()

def remove_extra_newlines(s):
    """
    Removes extra newlines from strings
    """
    return re.sub("\n+","\n",s)

def remove_extra_tabs(s):
    """
    Removes extra tabs from strings
    """
    return re.sub("\t+","\t",s)

def remove_extra_spaces(s):
    """
    Removes extra spaces from strings
    """
    s = remove_extra_whitespaces(s)
    s = remove_extra_newlines(s)
    s = remove_extra_tabs(s)
    return s

def split_after_colon(text,max_left=10):
    """
    Splits text after every colon.
    """
    splits = re.split("ঃ|:",text)
    if len(splits)==2 and len(splits[0])<=max_left:
        text = splits[-1]

    return text.strip()

def lowercase(text):
    """Converts all characters to lowercase"""
    if isinstance(text,str):
        return text.lower()
    else:
        return text

def remove_quotes(string):
    """
    Removes all quotes Single(') or Double(") from strings 
    """
    pattern = re.compile("\'|\"")
    return re.sub(pattern,'',string)

def remove_special_chars(s):
    """
    Removes the following special characters from strings
    """
    s = re.sub('[\\\/!,*)@#%(&$_.?^-]', ' ', s)
    s = remove_quotes(s)
    return re.sub(" +"," ",s).strip()

def remove_space_between_numbers(text):
    """
    Removes whitespaces between numbers
    """
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text


def remove_dashes(x):
    x = re.sub("-+", " ", x)
    x = re.sub(" +", " ", x)
    return x

def remove_commas_between_numbers(text):
    """
    Removes commas between numbers
    """
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1\2', text)
    return text

def replace_numbers(text):
    """
    Replaces all numbers with a given string
    """
    return re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), text)

def keep_alnum_only(text):
    """
    Removes all non-alphanumeric characters from strings
        NOTE: This function will remove whitespaces as well.
    """
    return ''.join([a for a in text if a.isalnum()])

def keep_whitespace_alnum(text):
    """
    Removes all non-alphanumeric characters from strings
    Keeps whitespaces, and alpha-numeric characters
    """
    if isinstance(text,str):
        text = ''.join([a if a.isalnum() else ' ' for a in text])
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
    digit_counts = list(np.unique([(i.end()-i.start()) for i in list(re.finditer("#+",s))]))
    digit_counts.sort()
    digit_counts.reverse()
    for d in digit_counts:
        s = re.sub(''.join(['#' for _ in range(d)]),f" {d}# ",s)    
    return re.sub(" +",' ',s).strip()

def remove_stopwords(text:str,stopwords):
    """
    Removes all stopwords from strings
    """
    return " ".join([t for t in text.split() if t not in stopwords])

def remove_short_words(text:str,min_length=2):
    """
    Removes all words with length less than min_length
    """
    return " ".join([t for t in text.split() if len(t)>=min_length])

def replace_whole_words(text,replace_words:dict):
    """
    Replaces certain words with a given string
    For example:
        'Yes' is often incorrectly tagged as 'G' in the dataset
    So we replace all 'G'/'G.' with 'Yes'
    """
    return " ".join([replace_words[t] if t in replace_words else t for t in text.split()])


'''
#contractions_path = os.path.join(package_dir,r'assets/contractions.json')
#stopwords_path = os.path.join(package_dir,r'assets/stopwords.json')
#contractions = json.load(open(contractions_path,'r'))
#stopwords = json.load(open(stopwords_path ,'r'))

def remove_transliteration_accents(text):

    """Removes transliteration accents from strings"""    
    nfkd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nfkd_form if (unicodedata.combining(c)==0) and (unicodedata.name(c).startswith('BEN')==False)])

def translate_numbers(s,bengali_num_dict):
    """
    For bengali text only.
    Replaces all numbers with their respective bengali numbers.

    Dictionary not in repository.
    """
    for n in range(100,-1,-1):
        s = re.sub(str(n),bengali_num_dict[str(n)],s)
    return s

def replace_contractions(text,contractions):
    """
    Replaces all contractions in strings
    """
    for c in contractions:
        # Replace full words
        text = re.sub(r"\b{}\b".format(c),contractions[c],text,flags=re.IGNORECASE)
    return text

        
    
'''

