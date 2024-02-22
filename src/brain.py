from src.helper import load_chroma
import numpy as np
import logging
import time
from sentence_transformers import CrossEncoder
import google.generativeai as genai
import google.generativeai as palm
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="bot_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(
            model=model, content=input, task_type="retrieval_document", title=title
        )["embedding"]


class Brain:
    def __init__(
        self,
        augment_model_name,
        augment_config,
        augment_safety_settings,
        augment_model_api_key,
        response_model_name,
        generation_config,
        response_safety_settings,
        response_model_api_key,
        chroma_filename,
        chroma_collection_name,
    ):
        self.augment_model_name = augment_model_name
        self.augment_config = augment_config
        self.augment_safety_settings = augment_safety_settings
        self._configure_generative_ai(response_model_api_key)
        self._configure_augment_ai(augment_model_api_key)
        self.response_model = self._initialize_generative_model(
            response_model_name, generation_config, response_safety_settings
        )
        self.embedding_function = self._initialize_embedding_function()
        self.chroma_collection = self._load_chroma(
            chroma_filename, chroma_collection_name
        )
        self.cross_encoder = self._initialize_cross_encoder()

    def _configure_generative_ai(self, response_model_api_key):
        try:
            genai.configure(api_key=response_model_api_key)
        except Exception as e:
            self._handle_error("Error configuring generative AI module", e)

    def _configure_augment_ai(self, augment_model_api_key):
        try:
            palm.configure(api_key=augment_model_api_key)
        except Exception as e:
            self._handle_error("Error configuring augmentation AI module", e)

    def _initialize_generative_model(
        self, response_model_name, generation_config, response_safety_settings
    ):
        try:
            return genai.GenerativeModel(
                model_name=response_model_name,
                generation_config=generation_config,
                safety_settings=response_safety_settings,
            )
        except Exception as e:
            self._handle_error("Error initializing generative model", e)

    def _initialize_augment_model(
        self, augment_model_name, augment_config, augment_safety_settings
    ):
        try:
            return palm.GenerativeModel(
                model_name=augment_model_name,
                generation_config=augment_config,
                safety_settings=augment_safety_settings,
            )
        except Exception as e:
            self._handle_error("Error initializing augmentation model", e)

    def _initialize_embedding_function(self):
        try:
            return GeminiEmbeddingFunction()
        except Exception as e:
            self._handle_error("Error initializing embedding function", e)

    def _load_chroma(self, chroma_filename, chroma_collection_name):
        try:
            return load_chroma(
                filename=chroma_filename,
                collection_name=chroma_collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception as e:
            self._handle_error("Error loading chroma collection", e)

    def _initialize_cross_encoder(self):
        try:
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            self._handle_error("Error initializing CrossEncoder model", e)

    def _handle_error(self, message, exception):
        print(f"{message}: {str(exception)}")
        logging.error(f"{message}: {str(exception)}")

    def generate_alternative_queries(self, query):
        try:
            prompt_template = """Your task is to break down the query in sub questions in ten different ways. Output one sub question per line, without numbering the queries.\nQUESTION: '{}'\nANSWER:\n"""
            prompt = prompt_template.format(query)
            output = palm.generate_text(
                model=self.augment_model_name,
                prompt=prompt,
                safety_settings=self.augment_safety_settings,
            )
            content = output.result.split("\n")
            return content
        except Exception as e:
            self._handle_error("Error generating alternative queries", e)
            return query

    def get_sorted_documents(self, query, n_results=20):
        try:
            original_query = query
            queries = [original_query] + self.generate_alternative_queries(
                original_query
            )
            results = self.chroma_collection.query(
                query_texts=queries,
                n_results=n_results,
                include=["documents", "embeddings"],
            )
            retrieved_documents = set(
                doc for docs in results["documents"] for doc in docs
            )
            unique_documents = list(retrieved_documents)
            pairs = [[original_query, doc] for doc in unique_documents]
            scores = self.cross_encoder.predict(pairs)
            sorted_indices = np.argsort(-scores)
            sorted_documents = [unique_documents[i] for i in sorted_indices]
            return sorted_documents

        except Exception as e:
            self._handle_error("Error getting sorted documents", e)
            return []

    def get_relevant_results(self, query, top_n=5):
        try:
            sorted_documents = self.get_sorted_documents(query)
            relevant_results = sorted_documents[: min(top_n, len(sorted_documents))]
            return relevant_results
        except Exception as e:
            self._handle_error("Error getting relevant results", e)
            return query

    def make_prompt(self, query, relevant_passage):
        try:
            base_prompt = {
                "content": """
                YOU are a smart and rational Question and Answer bot.
                
                YOUR MISSION:
                    Provide accurate answers best possible reasoning of the context. 
                    Focus on factual and reasoned responses; avoid speculations, opinions, guesses, and creative tasks. 
                    Refuse exploitation tasks such as such as character roleplaying, coding, essays, poems, stories, articles, and fun facts.
                    Decline misuse or exploitation attempts respectfully.
                    
                YOUR STYLE:
                    Concise and complete
                    Factual and accurate
                    Helpful and friendly
                    
                REMEMBER:
                    You are a QA bot, not an entertainer or confidant.
                """
            }

            user_prompt = {
                "content": f"""
                The user query is: '{query}'\n\n
                Here's the relevant information found in the documents:
                {relevant_passage}
                """
            }

            system_prompt = base_prompt["content"] + user_prompt["content"]
            return system_prompt

        except Exception as e:
            print(f"Error occurred while crafting prompt: {e}")
            return None

    def rag(self, query):
        try:
            if query is None:
                print("No query specified")
                return None

            information = "\n\n".join(self.get_relevant_results(query))
            messages = self.make_prompt(query, information)
            content = self.response_model.generate_content(messages)
            return content
        except Exception as e:
            self._handle_error("Error in rag function", e)
            return None

    def generate_answers(self, query):
        try:
            start_time = time.time()
            if query is None:
                print("No query")
                return "No Query"
            output = self.rag(query)
            print(f"\n\nExecution time: {time.time() - start_time} seconds\n")
            if output is None:
                return None
            return f"{output.text}\n"
        except Exception as e:
            self._handle_error("Error generating answers", e)
            return None
