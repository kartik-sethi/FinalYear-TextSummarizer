import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    # Remove full stops, commas, double quotes, and other symbols
    text = re.sub(r'[.,\"\'!@#$%^&*()_+={}\[\]:;<>,?/~`\\]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
  
    
    # Join the processed words back into a string
    processed_text = " ".join(words)
    
    return processed_text

# Example usage
input_text = "This is an example sentence. It includes punctuation, stopwords, and different verb forms."
processed_text = preprocess_text(input_text)


def clean_text(s):
    s = s.lower()
    s = s.split()
    s = " ".join(s)
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)
    return s

def remove_stop_words(s):
    stop_words = set(stopwords.words('english'))
    s = s.split()
    s = [w for w in s if not w.lower() in stop_words]
    s = " ".join(s)
    return s

def extract_topics(text):
    if not text or text.isspace():
        return [] 

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Tokenize the preprocessed text
    words = word_tokenize(preprocessed_text)
    
    tfv = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None,
        stop_words='english',  # Exclude common English stop words
        max_df=0.85,  # Ignore words that occur in more than 85% of documents
    )
    corpus_transformed = tfv.fit_transform(words)
    
    svd = TruncatedSVD(n_components=5)
    corpus_svd = svd.fit(corpus_transformed)
    
    feature_scores = dict(
        zip(
            tfv.get_feature_names_out(),
            corpus_svd.components_[0]
        )
    )

    topic_output = sorted(
        feature_scores, key=feature_scores.get, reverse=True
    )[:5]

    return topic_output
 

# Input text
input_text = """
In the heart of the bustling city, where the vibrant energy of urban life meets the soothing rhythm of tradition, 
there exists a unique blend of cultures and stories. Streets lined with towering skyscrapers intertwine with historic landmarks 
that whisper tales of bygone eras. The diverse tapestry of people, each carrying their dreams and aspirations, contributes to 
the colorful mosaic of this metropolis. From the enchanting aroma of street food wafting through narrow alleyways to the symphony 
of languages spoken in lively marketplaces, every corner resonates with life. As day turns to night, the city transforms into 
a spectacle of lights, casting a magical glow on its skyline. Amidst the constant motion and the ebb and flow of daily life, 
there is an undeniable sense of unity that binds the residents together, creating a sense of belonging in this ever-evolving 
urban landscape.
"""

# Call the extract_topics function with the input text
topics = extract_topics(input_text)

# Print the output
print("Top 5 topics:")
for i, topic in enumerate(topics, start=1):
    print(f"{i}. {topic}")
