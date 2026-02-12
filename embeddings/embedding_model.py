import logging
from io import BytesIO
from typing import List, Optional

import clip
import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ProductEmbeddingModel:
    """CLIP-based embedding model for images and text.

    Uses OpenAI CLIP ViT-B/32 to produce 512-dim embeddings suitable
    for cosine-similarity search.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Loaded CLIP %s on %s", model_name, self.device)

    # ------------------------------------------------------------------
    # Single-item embeddings
    # ------------------------------------------------------------------

    def get_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """Return a (1, 512) embedding for an image URL, or *None* on failure."""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(tensor)

            return features.cpu().numpy()

        except Exception as e:
            logger.warning("Failed to embed image %s: %s", image_url, e)
            return None

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Return a (1, 512) embedding for a text query, or *None* on failure."""
        try:
            tokens = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                features = self.model.encode_text(tokens)

            return features.cpu().numpy()

        except Exception as e:
            logger.warning("Failed to embed text: %s", e)
            return None

    # ------------------------------------------------------------------
    # Batch embeddings
    # ------------------------------------------------------------------

    def get_image_embeddings_batch(
        self, image_urls: List[str], batch_size: int = 32
    ) -> tuple[np.ndarray, List[int]]:
        """Embed many images at once, returning stacked features and valid indices.

        Returns:
            embeddings: np.ndarray of shape (N, 512) for the successfully embedded images.
            valid_indices: list of indices (into *image_urls*) that succeeded.
        """
        all_features: List[np.ndarray] = []
        valid_indices: List[int] = []
        batch_tensors: List[torch.Tensor] = []
        batch_indices: List[int] = []

        for i, url in enumerate(image_urls):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                batch_tensors.append(self.preprocess(img))
                batch_indices.append(i)
            except Exception as e:
                logger.warning("Skipping image %s: %s", url, e)
                continue

            # Flush batch when full
            if len(batch_tensors) >= batch_size:
                feats = self._encode_image_batch(batch_tensors)
                all_features.append(feats)
                valid_indices.extend(batch_indices)
                batch_tensors, batch_indices = [], []

        # Flush remaining
        if batch_tensors:
            feats = self._encode_image_batch(batch_tensors)
            all_features.append(feats)
            valid_indices.extend(batch_indices)

        if not all_features:
            return np.empty((0, 512), dtype=np.float32), []

        return np.concatenate(all_features, axis=0), valid_indices

    def _encode_image_batch(self, tensors: List[torch.Tensor]) -> np.ndarray:
        """Encode a list of preprocessed image tensors."""
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
        return features.cpu().numpy()


