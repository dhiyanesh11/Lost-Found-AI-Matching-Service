from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Lost & Found AI Matching Service")

class Item(BaseModel):
    id: str
    title: str
    description: str
    location: str

class MatchRequest(BaseModel):
    found_item: Item
    lost_items: List[Item]
    threshold: float = 0.6

@app.post("/match")
def match_items(data: MatchRequest):

    found_text = f"{data.found_item.title} {data.found_item.description} {data.found_item.location}"

    lost_texts = [
        f"{item.title} {item.description} {item.location}"
        for item in data.lost_items
    ]

    corpus = [found_text] + lost_texts

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    results = []
    for idx, score in enumerate(similarities[0]):
        if score >= data.threshold:
            results.append({
                "lostItemId": data.lost_items[idx].id,
                "similarity": float(score)
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "matches": results,
        "total_matches": len(results)
    }