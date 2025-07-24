import re
import time
import nltk
import torch
import string
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")


class NewsCatPredictPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = r"models\fake_news_classifier\classification_model"
        self.tokenizer_path = r"models\fake_news_classifier\classification_tokenizer"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=2).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.pipe = pipeline('text-classification',
                             model=self.model, tokenizer=self.tokenizer,
                             device=self.device, max_length=128, truncation=True)

    def clean_text_for_nlp(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = re.sub(r'\badvertisement\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(read more|related story|subscribe now)\b', '', text,
                      flags=re.IGNORECASE)
        words = text.split()
        filtered_words = []
        for word in words:
            if word not in stopwords.words('english'):
                filtered_words.append(word)
        text = " ".join(filtered_words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text):
        cleaned_text = self.clean_text_for_nlp(text)
        result = self.pipe(cleaned_text)
        return result[0]['label'], result[0]['score']

if __name__ == "__main__":
    predictor = NewsCatPredictPipeline()
    sample_text = "The Supreme Court of the United States has issued a sweeping series of rulings that significantly limit the power of federal agencies, marking one of the most consequential shifts in administrative law in decades. In a coordinated release of thirteen decisions, the Court redefined the scope of executive authority, particularly targeting how agencies interpret and enforce regulations without direct congressional approval. These rulings affect a wide range of sectors, including environmental protection, labor standards, and consumer rights, and are expected to trigger a wave of legal challenges against long-standing federal policies Legal scholars and political analysts are calling this a landmark moment in the ongoing debate over the “administrative state.” The Court’s conservative majority argued that unelected bureaucrats should not have broad discretion to shape national policy, asserting that such powers must be clearly delegated by Congress. Critics, however, warn that the decisions could paralyze regulatory enforcement and leave critical issues—such as climate change, workplace safety, and financial oversight—vulnerable to political gridlock. The Biden administration expressed deep concern over the implications of the rulings, stating that they could undermine the federal government’s ability to respond swiftly to emerging crises and protect public welfare. Progressive lawmakers have already begun drafting legislation aimed at restoring some of the agencies’ lost authority, though passage in a divided Congress remains uncertain. Meanwhile, conservative groups and business coalitions have celebrated the decisions as a victory for constitutional governance and economic freedom. They argue that the rulings will reduce regulatory burdens and promote transparency and accountability in federal rulemaking. As the dust settles, legal experts anticipate years of litigation and legislative battles as the country adjusts to a dramatically altered regulatory landscape."
    start = time.time()
    label, score = predictor.predict(sample_text)
    end = time.time()
    print(f"Predicted Label: {label}, Score: {score:.4f}, Time taken: {end - start:.2f} seconds")