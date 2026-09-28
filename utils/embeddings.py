# # utils/embeddings.py

# import google.generativeai as genai
# import os
# import numpy as np
# import json
# from typing import List, Dict, Any, Union
# import time

# class EmbeddingModel:
#     """
#     A wrapper for Google's Gemini Embedding models.
#     Handles embedding single queries and batches of documents.
#     """
#     def __init__(self, model_name: str = "models/text-embedding-004"):
#         """
#         Initializes the EmbeddingModel.

#         Args:
#             model_name (str): The name of the embedding model to use.
#         """
#         if not os.getenv("GOOGLE_API_KEY"):
#             raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Gemini API key.")
        
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         self.model_name = model_name
        
#         print(f"[EmbeddingModel] Initialized with model: {self.model_name}")

#     def _debug_response(self, response):
#         """Debug helper to understand response structure."""
#         print(f"[DEBUG] Response type: {type(response)}")
#         print(f"[DEBUG] Response attributes: {dir(response)}")
#         if hasattr(response, '__dict__'):
#             print(f"[DEBUG] Response dict: {response.__dict__}")

#     def _extract_embeddings_from_response(self, response: Any) -> List[List[float]]:
#         """
#         Helper to safely extract embeddings from the API response.
#         Updated for current Google AI SDK.
#         """
#         extracted_embeddings = []
        
#         try:
#             # Debug the response structure
#             # self._debug_response(response)
            
#             # New SDK structure: response.embedding for single, response.embeddings for batch
#             if hasattr(response, 'embedding') and response.embedding:
#                 # Single embedding - try different attribute names
#                 embedding_data = response.embedding
#                 if hasattr(embedding_data, 'values') and embedding_data.values:
#                     extracted_embeddings.append(list(embedding_data.values))
#                 elif isinstance(embedding_data, (list, np.ndarray)):
#                     extracted_embeddings.append(list(embedding_data))
#                 else:
#                     print(f"[EmbeddingModel] Unknown embedding format: {type(embedding_data)}")
                    
#             elif hasattr(response, 'embeddings') and response.embeddings:
#                 # Multiple embeddings
#                 for item in response.embeddings:
#                     if hasattr(item, 'values') and item.values:
#                         extracted_embeddings.append(list(item.values))
#                     elif isinstance(item, (list, np.ndarray)):
#                         extracted_embeddings.append(list(item))
                        
#             # Handle dictionary response (error cases or alternative format)
#             elif isinstance(response, dict):
#                 if 'embedding' in response and response['embedding']:
#                     if isinstance(response['embedding'], (list, np.ndarray)):
#                         extracted_embeddings.append(list(response['embedding']))
#                 elif 'embeddings' in response and response['embeddings']:
#                     for item in response['embeddings']:
#                         if isinstance(item, (list, np.ndarray)):
#                             extracted_embeddings.append(list(item))
#             else:
#                 print(f"[EmbeddingModel] Unable to extract embeddings from response structure")
#                 self._debug_response(response)
                
#         except Exception as e:
#             print(f"[EmbeddingModel] Error during embedding extraction: {e}")
#             self._debug_response(response)
        
#         return extracted_embeddings

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """
#         Embeds a list of text documents.
#         """
#         if not texts:
#             return []
            
#         # Filter out empty texts
#         valid_texts = [text.strip() for text in texts if text and text.strip()]
#         if not valid_texts:
#             print("[EmbeddingModel] No valid texts to embed")
#             return []
            
#         try:
#             # Process one by one to avoid batch issues
#             all_embeddings = []
            
#             for i, text in enumerate(valid_texts):
#                 try:
#                     response = genai.embed_content(
#                         model=self.model_name,
#                         content=text,
#                         task_type="RETRIEVAL_DOCUMENT"
#                     )
                    
#                     batch_embeddings = self._extract_embeddings_from_response(response)
#                     all_embeddings.extend(batch_embeddings)
                    
#                     # Small delay to respect rate limits
#                     if i < len(valid_texts) - 1:
#                         time.sleep(0.1)
                        
#                 except Exception as e:
#                     print(f"[EmbeddingModel] Error embedding document {i}: {e}")
#                     continue
            
#             print(f"[EmbeddingModel] Successfully extracted {len(all_embeddings)} document embeddings.")
#             return all_embeddings
            
#         except Exception as e:
#             print(f"[EmbeddingModel] Error in embed_documents: {e}")
#             return []

#     def embed_query(self, text: str) -> List[float]:
#         """
#         Embeds a single query text.
#         """
#         if not text or not text.strip():
#             print("[EmbeddingModel] Empty query text provided")
#             return []
            
#         try:
#             response = genai.embed_content(
#                 model=self.model_name,
#                 content=text.strip(),
#                 task_type="RETRIEVAL_QUERY"
#             )
            
#             embeddings = self._extract_embeddings_from_response(response)
            
#             if embeddings and len(embeddings) > 0:
#                 embedding = embeddings[0]  # Take first (and should be only) embedding
#                 print(f"[EmbeddingModel] Successfully generated query embedding of dimension {len(embedding)}")
#                 return embedding
#             else:
#                 print(f"[EmbeddingModel] No embedding found in response for query: '{text[:50]}...'")
#                 return []
                
#         except Exception as e:
#             print(f"[EmbeddingModel] Error in embed_query for '{text[:50]}...': {e}")
#             return []

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Local embedding model wrapper used by the RAG pipeline.

    Uses Sentence Transformers instead of a cloud embedding API,
    so document and query embeddings can be generated without
    requiring an API key.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.embedding_dimension = self.model.get_embedding_dimension()

        print(
            f"[EmbeddingModel] Initialized with model: {self.model_name} "
            f"(dimension={self.embedding_dimension})"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for document chunks.
        """
        if not texts:
            return []

        valid_texts = [
            text.strip()
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not valid_texts:
            return []

        try:
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return embeddings.tolist()

        except Exception as exc:
            print(f"[EmbeddingModel] Error embedding documents: {exc}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a user query.
        """
        if not text or not text.strip():
            return []

        try:
            embedding = self.model.encode(
                text.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return embedding.tolist()

        except Exception as exc:
            print(f"[EmbeddingModel] Error embedding query: {exc}")
            return []