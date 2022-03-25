import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def removeStopWords(txt):
    textarray = txt.split('|')
    for i, textprompt in enumerate(textarray):
        print(len(textprompt))
        if len(textprompt) > 77:
            # Cut it off
            text_tokens = word_tokenize(txt)
            tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
            filtered_sentence = (" ").join(tokens_without_sw)
            textarray[i] = filtered_sentence
    fixedtext = (" ").join(textarray)
    return fixedtext