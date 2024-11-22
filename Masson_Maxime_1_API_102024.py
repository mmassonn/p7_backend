# importer le module
from flask import Flask, request, render_template, jsonify
from huggingface_hub import login, hf_hub_download
import torch


# créer l'object application
app = Flask(__name__, template_folder='templates')

login(token="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")

tokenizer_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="tokenizer.pth")
model_file = hf_hub_download(repo_id="mmassonn/Badbuzzert", filename="bert_model.pth")

# Charger le tokenizer et le modèle
tokenizer = torch.load(tokenizer_file, map_location=torch.device('cpu'))
model = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)

@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"

# spécifier les process de l'API
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Tokenisation
    encoded_input = tokenizer(text, return_tensors='pt')

    # Prédiction
    outputs = model(**encoded_input)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    # Mapping de l'ID de classe vers une étiquette (à adapter selon votre modèle)
    labels = {
        0: "Le sentiment de ce tweet est negatif", 
        1: "Le sentiment de ce tweet est positif"
        }

    predicted_label = labels[predicted_class.item()]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)