import os
from src.brain import Brain
import re

class Initializer:
    @staticmethod
    def initialize(augment_api_key, response_api_key, chroma_filename):
        response_model_name = "gemini-pro"
        augment_model_name = "models/text-bison-001"
        generation_config = {
            "temperature": 0.9,
            "top_p": 0.7,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        response_safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        augment_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 80,
            "max_output_tokens": 1024,
        }
        augment_safety_settings = [
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 4},
            {"category": "HARM_CATEGORY_TOXICITY", "threshold": 4},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 4},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 4},
            {"category": "HARM_CATEGORY_MEDICAL", "threshold": 4},
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 4},
        ]
        def base_name(file_path):
            base_name = os.path.basename(file_path)
            name, extension = os.path.splitext(base_name)
            return name 
        def clean_up(message):
            message = re.sub(r"[^\w\s,]", "", message)
            message = re.sub(r"http\S+|www.\S+", "", message)
            message = re.sub(r"\s+", "", message)
            return message[:30]

        chroma_collection_name = str.upper(clean_up(base_name(chroma_filename))) + "_COLLECT"

        return Brain(
            augment_model_name,
            augment_config,
            augment_safety_settings,
            augment_api_key,
            response_model_name,
            generation_config,
            response_safety_settings,
            response_api_key,
            chroma_filename,
            chroma_collection_name,
        )
    
