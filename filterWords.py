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
            if len(filtered_sentence) > 77:
                filtered_sentence = filtered_sentence[0:60]
            textarray[i] = filtered_sentence
    fixedtext = (" ").join(textarray)

    return fixedtext