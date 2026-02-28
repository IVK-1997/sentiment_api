from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI()

client = OpenAI()


class CommentRequest(BaseModel):
    comment: str


class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment classifier.\n"
                        "Return ONLY valid JSON in this exact format:\n"
                        '{"sentiment": "positive|negative|neutral", "rating": integer_between_1_and_5}\n'
                        "Do not include any extra text."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ]
        )

        raw_output = response.output_text

        parsed = json.loads(raw_output)

        if parsed["sentiment"] not in ["positive", "negative", "neutral"]:
            raise ValueError("Invalid sentiment")

        if not (1 <= int(parsed["rating"]) <= 5):
            raise ValueError("Invalid rating")

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
