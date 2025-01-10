# importer le module
from flask import Flask, request, jsonify
from huggingface_hub import login, hf_hub_download
import xgboost as xgb
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from joblib import load

# Créer l'object application
app = Flask(__name__, template_folder='templates')

# Remplacez par vos clés d'API Hugging Face et les noms de votre tokenizer et modèle
login(token="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")

#  Charger le tfidf_vectorizer
tfidf_vectorizer_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="tfidf_vectorizer.joblib")
vectorizer = load(tfidf_vectorizer_file)
#  Charger le modèle
model_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="xgb_model.model")
model = xgb.XGBClassifier()
model.load_model(model_file)

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

@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"

# spécifier les process de l'API
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Tokenisation
    text_preprocessed = preprocess_text(text)
    text_preprocessed = [text_preprocessed]
    X_test_lr = vectorizer.transform(text_preprocessed)
    y_preds = model.predict(X_test_lr.toarray())
    # Mapping de l'ID de classe vers une étiquette (à adapter selon votre modèle)
    labels = {
        0: "Le sentiment de ce tweet est negatif", 
        1: "Le sentiment de ce tweet est positif"
        }

    predicted_label = labels[y_preds[0]]

    return jsonify({'prediction': predicted_label})

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
import logger
from http_exceptions import HTTPException

# Configurer Azure Monitor pour OpenTelemetry
INSTRUMENTATION_KEY = "84a4c70c-82e5-44ea-852e-46d0a5097c58"
#configure_azure_monitor(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}")

# Obtenir un tracer pour générer des spans
tracer = trace.get_tracer(__name__)

@app.post("/log_trace")
def log_trace():
    data = request.get_json(force=True)
    text = data['text']
    predicted_sentiment = data['predicted_sentiment']
    """
    Endpoint pour enregistrer une trace en cas de prédiction incorrecte.
    """
    try:
        with tracer.start_as_current_span("PredictionErrorTrace") as span:
            # Ajouter les attributs au span
            span.set_attribute("event.type", "prediction_incorrect")
            span.set_attribute("text", text)
            span.set_attribute("predicted_sentiment", predicted_sentiment)
            span.set_attribute("message", "Prédiction signalée comme incorrecte par l'utilisateur.")

            # Log dans la console pour confirmation
            logger.warning(f"Prédiction incorrecte signalée : {text} "
                           f"(Sentiment : {predicted_sentiment})")

        return {"message": "Trace enregistrée avec succès dans Azure Application Insight"}

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement de la trace : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'enregistrement de la trace.")


if __name__ == '__main__':
    app.run(debug=True)
