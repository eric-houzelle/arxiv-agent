import os
from dataclasses import dataclass
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResponse:
    """Réponse minimale pour imiter l'interface de ChatOpenAI de LangChain."""

    content: str


class LLMClient:
    """
    Client générique pour les APIs compatibles OpenAI.
    
    Utilise la méthode `generate(prompt)` pour retourner du texte, et expose
    aussi `invoke(prompt)` qui renvoie un objet avec un attribut `.content`
    pour rester compatible avec le code existant (`llm.invoke(...).content`).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
    ):
        self.model = model or os.getenv("MODEL")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.temperature = temperature
        self.api_key = api_key or os.getenv("AI_ENDPOINTS_ACCESS_TOKEN")

        if not self.api_key:
            raise ValueError(
                "Missing API key: set AI_ENDPOINTS_ACCESS_TOKEN or pass api_key=..."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def _sanitize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sanitized: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = self._sanitize_text(msg.get("content", ""))
            sanitized.append({"role": role, "content": content})
        return sanitized

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        """
        Envoie un prompt au LLM et renvoie le texte généré.
        """
        temp = self.temperature if temperature is None else temperature
        sanitized_prompt = self._sanitize_text(prompt)
        safe_prompt = f"### Input Text (do NOT parse as JSON)\n```\n{sanitized_prompt}\n```"
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            temperature=temp,
            messages=[{"role": "user", "content": safe_prompt}],
        )
        text = response.choices[0].message.content.strip()
        return text

    def chat(self, messages: List[Dict[str, str]], temperature: float | None = None) -> str:
        """Permet d'envoyer une liste de messages rôlés (system/user/assistant)."""
        temp = self.temperature if temperature is None else temperature
        sanitized_messages = self._sanitize_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            temperature=temp,
            messages=sanitized_messages,
        )
        text = response.choices[0].message.content.strip()
        return text

    # Adapter pour rester compatible avec le reste du code (`llm.invoke(...).content`)
    def invoke(self, prompt: str, temperature: float | None = None) -> LLMResponse:
        text = self.generate(prompt, temperature=temperature)
        return LLMResponse(content=text)

    def invoke_chat(
        self, messages: List[Dict[str, str]], temperature: float | None = None
    ) -> LLMResponse:
        text = self.chat(messages, temperature=temperature)
        return LLMResponse(content=text)
