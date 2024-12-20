from joblib import load
from huggingface_hub import login, hf_hub_download
import pytest
import xgboost as xgb
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

@pytest.fixture

# Teste le chargement du vectoriseur.
def test_model_loading():
    login(token="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")
    tfidf_vectorizer_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="tfidf_vectorizer.joblib")
    vectorizer = load(tfidf_vectorizer_file)
    # Vérifie que le modèle a été chargé correctement
    assert vectorizer is not None, "Erreur dans le chargement du vectoriseur."

# Teste le chargement du modèle.
def test_model_loading():
    login(token="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")
    model_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="xgb_model.model")
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    # Vérifie que le modèle a été chargé correctement
    assert model is not None, "Erreur dans le chargement du modèle."

# Prétraitement

# Definir le dictionnaire contenant les émojis avec leurs significations.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Definir tous les stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
final_stopwords = list(stopwords.words('english'))+stopwordlist

# Créer un Lemmatiseur et un Stemming.
lemmatizer = WordNetLemmatizer()

# Définir des modèles d'expressions régulières.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

def preprocess_text(tweet: str,
                    stem_or_lem: str = "lem"
                    ) -> str:
    """
    Cette fonction applique le prétraitement à un tweet :
      1* Si le tweet est une liste, concaténe en une seule chaine.
      2* Convertir le tweet en minuscule.
      3* Remplacez toutes les URL par «URL».
      4* Remplacer tous les emojis par un mot.
      5* Remplacer @USERNAME par 'USER'.
      6* Remplacer tous les caractères non alphabets par un espace.
      7* Remplacer 3 lettres consécutives ou plus par 2 lettres.
      8* Vérifier si le mot est dans stopwordlist.
      9* Lemmatiser ou stemmed le mot si pas dans stopword.
    Args:
      tweet (str): Texte du tweet à prétraiter.
      stem_or_lem (str): Valeurs possibles "lem"/"stem". Permet de choisir si
      le texte doit être stemmé ou lemmatisé.
    Returns:
      tweetwords (str): Texte après le prétraitement
    """

    if type(tweet) == list:
        tweet = " ".join(tweet)
    # Convertir le tweet en minuscule.
    tweet = tweet.lower()
    # Remplacez toutes les URL par «URL».
    tweet = re.sub(urlPattern,"URL",tweet)
    # Remplacer tous les emojis par un mot.
    for emoji in emojis.keys():
        tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
    # Remplacer @USERNAME par 'USER'.
    tweet = re.sub(userPattern,' USER', tweet)
    # Remplacer tous les caractères non alphabets par un espace.
    tweet = re.sub(alphaPattern, " ", tweet)
    # Remplacer 3 lettres consécutives ou plus par 2 lettres.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    tweetwords = ''
    for word in tweet.split():
        # Vérifier si le mot est dans stopwordlist:
        if len(word)>1 and word not in final_stopwords:
            # Lemmatizing le mot si pas dans stopword.
            if stem_or_lem == "lem":
                word = lemmatizer.lemmatize(word)
            tweetwords += (word+' ')

    return tweetwords

text = "I was so disappointed with this product. The quality is terrible and it broke after only a week. Don't waste your money!"

# Teste le pré-traitement des données
def test_pre_processing(text):
    text_preprocessed = preprocess_text(text)
    # Vérifie que le DataFrame n'est pas vide
    assert text_preprocessed is not None, "Erreur dans le pré-traitement."

# Teste la fonction de prédiction de l'API
def test_prediction(text):
    login(token="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")
    # Charger le Tokenizeur
    tfidf_vectorizer_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="tfidf_vectorizer.joblib")
    vectorizer = load(tfidf_vectorizer_file)
    # Charger le Modèle
    model_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="xgb_model.model")
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    # Tokenisation
    text_preprocessed = preprocess_text(text)
    text_preprocessed = [text_preprocessed]
    X_test_lr = vectorizer.transform(text_preprocessed)
    # Prédiction
    y_preds = model.predict(X_test_lr.toarray())
    assert y_preds is not None, "La prédiction a échoué."
