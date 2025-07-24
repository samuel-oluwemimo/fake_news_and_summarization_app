import time
from transformers import AutoTokenizer, pipeline

class SumPredictPipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(r"models\text_summarizer\tokenizer")
        self.model = r"models\text_summarizer\model_bbc_pegasus"
        self.pipe = pipeline('summarization', model=self.model, tokenizer=self.tokenizer)

    def predict(self, text):
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "min_length": 50,  # Set a minimum length
            "max_length": 256  # Increase the maximum length
        }
        result = self.pipe(text, **gen_kwargs)
        return result[0]['summary_text']

if __name__ == "__main__":
    predictor = SumPredictPipeline()
    sample_text = "Fyodor Dostoevsky’s Poor Folk is a deeply emotional and psychologically rich novel that explores the lives of two impoverished individuals, Makar Devushkin and Varvara Dobroselova, who maintain a tender and heartfelt correspondence through letters. Set in St. Petersburg, the story unfolds entirely through their written exchanges, revealing their inner worlds, struggles, and the harsh realities of poverty in 19th-century Russia. Makar, a low-ranking government clerk, is a kind but timid man who lives in a shabby boarding house and devotes much of his meager income to helping Varvara, a distant relative and the object of his quiet affection. Varvara, a young woman with a tragic past, is intelligent and sensitive but trapped by her circumstances and the limited options available to women of her class. Through their letters, they share memories, daily hardships, and dreams, creating a bond that is both touching and tragic. As the story progresses, the reader witnesses Makar’s increasing desperation and humiliation as he tries to maintain dignity in the face of social scorn and economic hardship, while Varvara becomes entangled with a manipulative benefactor who offers her a way out of poverty at the cost of her independence and emotional connection with Makar. The novel paints a poignant picture of the psychological toll of poverty, the yearning for human connection, and the quiet heroism of those who endure suffering with grace. Dostoevsky’s portrayal of these characters is filled with compassion and insight, making Poor Folk not just a social critique but also a profound exploration of human dignity, love, and sacrifice in the face of relentless adversity."
    start = time.time()
    summary = predictor.predict(sample_text)
    end = time.time()
    print(f"Summary: {summary}, Time taken: {end - start:.2f} seconds")