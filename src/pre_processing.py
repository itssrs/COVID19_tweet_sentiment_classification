import re
import string

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text): 
    url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)



def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
    
    return text2.lower()

def make_docs(data):
    docs = []
    for doc, label in nlp.pipe(data, as_tuples=True):
#         print("label is ",label,"doc is", doc)
        if label == 'Positive':
            doc.cats['Positive'] =  1
            doc.cats['Negative'] =  0
            doc.cats['Neutral']  =  0
        elif label == 'Negative':
            doc.cats['Positive'] =  0
            doc.cats['Negative'] =  1
            doc.cats['Neutral']  =  0
        else:
            doc.cats['Positive'] =  0
            doc.cats['Negative'] =  0
            doc.cats['Neutral']  =  1
#         print(doc.cats)
        docs.append(doc)
    return (docs)