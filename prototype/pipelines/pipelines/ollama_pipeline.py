from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import requests


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "ollama_pipeline"
        self.name = "Ollama Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.SYSTEM_PROMPT = """Jsi nápomocný chatbot Masarykovy univerzity. Tvým úkolem je pomáhat uživatelům orientovat se v Informačním systému (IS MU) a poskytovat rady, jak provést požadované akce v systému. Máš k dispozici oficiální dokumenty nápovědy IS MU, které mohou obsahovat užitečné informace."""
        pass # TODO refine system prompt, maybe in english since instructions

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def prepend_system_prompt(self, messages):
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}, *messages]
        else:
            messages[0]["content"] = self.SYSTEM_PROMPT
        return messages

    def decide_retrieve(self, body, messages):
        decision_messages = [
            {"role": "system", "content":
             "If you can answer directly, reply only: ANSWER"
             "If you need external info, reply only: RETRIEVE"
             "No other text."},
            *messages,
        ]
        decision_body = {
            **body,
            "messages":
            decision_messages,
            "model": self.MODEL,
            "stream": False
        }
        decision_resp = requests.post(
            f"{self.OLLAMA_BASE_URL}/v1/chat/completions",
            json=decision_body
        ).json()
        decision_text = decision_resp["choices"][0]["message"]["content"]
        retrieve = "RETRIEVE" in decision_text.upper()
        return retrieve

    def compose_query(self, body, messages):
        QUERY_PROMPT = """You are a query reformulator for a RAG system.
            TASK:
            Generate the best Czech search query based on the entire chat history.
            The goal is to retrieve relevant Czech documents.

            RULES:
            - Output only the query, nothing else.
            - The query must be in Czech.
            - Do not use pronouns like "it / this / that".
            - If the user's final question is unclear, infer the most relevant topic from context.
            - Maximum 1–2 sentences.

            OUTPUT:
            Only the Czech query text. No explanations.
            """
        query_messages = [
            {"role": "system", "content": QUERY_PROMPT},
            *messages
        ]

        query_body = {
            **body,
            "messages": query_messages,
            "model": self.MODEL,
            "stream": False
        }

        query = requests.post(
            self.OLLAMA_BASE_URL + "/v1/chat/completions",
            json=query_body
        ).json()["choices"][0]["message"]["content"]

        return query


    def retrieve(self, query):
        # TODO
        return "retrieval not implemented yet"


    def generate_with_retrieved(self, body, messages, retrieved):
        messages = [
            messages[0],
            {"role": "system", "content": f"Here are relevant parts from the official help pages (Nápověda):\n\n{retrieved}"},
            *messages[1:],
        ]
        r = requests.post(
            url=f"{self.OLLAMA_BASE_URL}/v1/chat/completions",
            json={**body, "messages":messages, "model": self.MODEL},
            stream=self.STREAM,
        )
        return r


    def _ollama_chat(self, body):
        r = requests.post(
            url=f"{self.OLLAMA_BASE_URL}/v1/chat/completions",
            json={**body, "model": self.MODEL},
            stream=self.STREAM,
        )

        r.raise_for_status()
        return r

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}", flush=True)

        self.OLLAMA_BASE_URL = "http://localhost:11434"
        self.MODEL = "qwen3:14b"
        self.STREAM = True


        if "user" in body:
            print("######################################", flush=True)
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})', flush=True)
            print(f"# Message: {user_message}", flush=True)
            print("######################################", flush=True)

        try:
            messages = self.prepend_system_prompt(messages)
            retrieve = self.decide_retrieve(body, messages)
            if retrieve:  
                query = self.compose_query(body, messages)
                retrieved = self.retrieve(query)
                r = self.generate_with_retrieved(body, messages, retrieved)
            else:
                r = self._ollama_chat(body)
            return r.iter_lines() if body["stream"] else r.json()

        except Exception as e:
            return f"Error: {e}"
