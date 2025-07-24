import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from ml.classification.predict import NewsCatPredictPipeline
from ml.summarization.predict import SumPredictPipeline
from pydantic import BaseModel


app = FastAPI()

# url = "https://www.nytimes.com/2025/06/11/business/elon-musk-trump-feud.html"
# details = extract_article_details(url)
# text = details['content']

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


# @app.get("/train")
# async def training():
#     try:
#         os.system("python main.py")
#         return Response("Training successful !!")
#
#     except Exception as e:
#         return Response(f"Error Occurred! {e}")


class TextInput(BaseModel):
    text: str

@app.post("/predict_cat")
async def predict_cat_route(input: TextInput):
    try:
        obj = NewsCatPredictPipeline()
        prediction = obj.predict(input.text)
        return {"category": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_sum")
async def predict_sum_route(input: TextInput):
    try:
        obj = SumPredictPipeline()
        prediction = obj.predict(input.text)
        return {"category": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
