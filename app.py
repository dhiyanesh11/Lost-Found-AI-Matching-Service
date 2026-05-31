from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Lost & Found - Unified AI Matching Service")

# ========================
#   LOAD MODELS (once at startup)
# ========================

# Text model: all-MiniLM-L6-v2 → 384-dim semantic embeddings
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Image model: MobileNetV2 → 1280-dim feature embeddings (14MB weights, low memory)
import torch
import torchvision.models as models
import torchvision.transforms as transforms

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.DEFAULT
    image_model = models.mobilenet_v2(weights=weights)
except ImportError:
    from torchvision.models import mobilenet_v2
    image_model = models.mobilenet_v2(pretrained=True)

image_model.classifier = torch.nn.Identity()
image_model.eval()

# Preprocessing transforms for MobileNetV2
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Thread pool for parallel encoding
executor = ThreadPoolExecutor(max_workers=2)


# ========================
#   REQUEST / RESPONSE MODELS
# ========================

class TextEmbedRequest(BaseModel):
    title: str
    description: str
    location: str

class ImageEmbedRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int

class BothEmbedRequest(BaseModel):
    title: str
    description: str
    location: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class BothEmbedResponse(BaseModel):
    textEmbedding: List[float]
    imageEmbedding: List[float]
    textDimensions: int
    imageDimensions: int

class StoredItem(BaseModel):
    id: str
    textEmbedding: Optional[List[float]] = None
    imageEmbedding: Optional[List[float]] = None

class MatchRequest(BaseModel):
    found_text_embedding: List[float]
    found_image_embedding: Optional[List[float]] = None
    lost_items: List[StoredItem]
    threshold: float = 0.6
    text_weight: float = 0.4
    image_weight: float = 0.6

class MatchResult(BaseModel):
    lostItemId: str
    similarity: float
    textSimilarity: float
    imageSimilarity: float


def _load_image(image_url: Optional[str], image_base64: Optional[str]) -> Image.Image:
    """Load image from base64 data or URL."""
    if image_base64:
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data)).convert("RGB")
    elif image_url:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise ValueError("Either image_url or image_base64 is required")


def _encode_text(title: str, description: str, location: str) -> list:
    """Generate text embedding."""
    combined_text = f"{title}. {description}. Lost near {location}"
    return text_model.encode(combined_text).tolist()


def _encode_image(image: Image.Image) -> list:
    """Generate image embedding using MobileNetV2."""
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = image_model(tensor)
    return features.squeeze(0).tolist()


@app.post("/embed/text", response_model=EmbeddingResponse)
def embed_text(data: TextEmbedRequest):
    """Generate a 384-dim semantic text embedding from item details."""
    embedding_list = _encode_text(data.title, data.description, data.location)
    return EmbeddingResponse(embedding=embedding_list, dimensions=len(embedding_list))


@app.post("/embed/image", response_model=EmbeddingResponse)
def embed_image(data: ImageEmbedRequest):
    """Generate a 512-dim semantic image embedding. Accepts base64 or URL."""
    image = _load_image(data.image_url, data.image_base64)
    embedding_list = _encode_image(image)
    return EmbeddingResponse(embedding=embedding_list, dimensions=len(embedding_list))


@app.post("/embed/both", response_model=BothEmbedResponse)
def embed_both(data: BothEmbedRequest):
    """
    Generate BOTH text and image embeddings in a single request.
    Runs text and image encoding in parallel using threads.
    This is ~2x faster than two separate calls.
    """
    # Load image first (I/O bound)
    image = _load_image(data.image_url, data.image_base64)

    # Run both encoders in parallel
    text_future = executor.submit(_encode_text, data.title, data.description, data.location)
    image_future = executor.submit(_encode_image, image)

    text_embedding = text_future.result()
    image_embedding = image_future.result()

    return BothEmbedResponse(
        textEmbedding=text_embedding,
        imageEmbedding=image_embedding,
        textDimensions=len(text_embedding),
        imageDimensions=len(image_embedding),
    )


@app.post("/match")
def match_items(data: MatchRequest):
    """
    Compare found item embeddings against all stored lost item embeddings.
    Combined score = text_weight × text_cosine + image_weight × image_cosine
    Returns matches above the threshold.
    """
    found_text = np.array(data.found_text_embedding).reshape(1, -1)
    has_found_image = data.found_image_embedding is not None and len(data.found_image_embedding) > 0

    if has_found_image:
        found_image = np.array(data.found_image_embedding).reshape(1, -1)

    results = []

    for item in data.lost_items:
        text_sim = 0.0
        image_sim = 0.0

        # Text similarity
        if item.textEmbedding and len(item.textEmbedding) > 0:
            lost_text = np.array(item.textEmbedding).reshape(1, -1)
            text_sim = float(cosine_similarity(found_text, lost_text)[0][0])
            # Clamp to [0, 1]
            text_sim = max(0.0, min(1.0, text_sim))

        # Image similarity
        if has_found_image and item.imageEmbedding and len(item.imageEmbedding) > 0:
            lost_image = np.array(item.imageEmbedding).reshape(1, -1)
            image_sim = float(cosine_similarity(found_image, lost_image)[0][0])
            image_sim = max(0.0, min(1.0, image_sim))

        # Combined score
        if has_found_image and item.imageEmbedding and len(item.imageEmbedding) > 0:
            # Both embeddings available → weighted combination
            combined = data.text_weight * text_sim + data.image_weight * image_sim
        else:
            # Only text available → use text score alone
            combined = text_sim

        if combined >= data.threshold:
            results.append(MatchResult(
                lostItemId=item.id,
                similarity=round(combined, 4),
                textSimilarity=round(text_sim, 4),
                imageSimilarity=round(image_sim, 4),
            ))

    # Sort by combined similarity descending
    results.sort(key=lambda x: x.similarity, reverse=True)

    return {
        "matches": [r.model_dump() for r in results],
        "total_matches": len(results)
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "text_model": "all-MiniLM-L6-v2",
        "text_dimensions": 384,
        "image_model": "mobilenet_v2",
        "image_dimensions": 1280,
    }