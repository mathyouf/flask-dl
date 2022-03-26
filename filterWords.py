import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import word_tokenize

nltk.download('punkt')



def removeStopWords(txt):
    textarray = txt.split('|')
    for i, textprompt in enumerate(textarray):
        print(len(textprompt))
        textarray[i] = textprompt[0:60]
    print(textarray)
    fixedtext = (" ").join(textarray)

    return fixedtext