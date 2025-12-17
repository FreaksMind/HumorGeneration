import pandas as pd
import spacy
from string import punctuation
import html
import re

df = pd.read_csv("one-million-reddit-jokes.csv", encoding='utf-8')

df = df[["id", "selftext"]]
df = df[df["selftext"] != "[removed]"]
df = df[df["selftext"] != "[deleted]"]
df["selftext"] = df["selftext"].fillna("")


# print(df)

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def get_keywords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

def get_top_keywords(text, n=2):
    keywords = get_keywords(text)
    keywords += [None]*(n - len(keywords))
    return keywords[:n]

df[["word1", "word2"]] = df["selftext"].apply(lambda x: pd.Series(get_top_keywords(x)))

# df.to_csv("jokes.csv", index=False, encoding="utf-8")

# df = pd.read_csv("jokes.csv")

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["selftext"] = df["selftext"].apply(clean_text)

df = df[df["word1"].notna() & df["word2"].notna() &
        (df["word1"].str.strip() != "") & (df["word2"].str.strip() != "")]

MIN_WORDS = 5
df = df[df["selftext"].str.split().str.len() >= MIN_WORDS]

from better_profanity import profanity

profanity.load_censor_words_from_file("profanity_words.txt")

mask = (
    df["word1"].fillna("").apply(profanity.contains_profanity) |
    df["word2"].fillna("").apply(profanity.contains_profanity)
)

df = df[~mask]

df.to_csv("jokes_clean.csv", index=False)
