import os
from dataclasses import dataclass

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

    def generate(self, prompt: str) -> str:
        """
        Envoie un prompt au LLM et renvoie le texte généré.
        """
        safe_prompt = f"### Input Text (do NOT parse as JSON)\n```\n{prompt}\n```"
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[{"role": "user", "content": safe_prompt}],
        )
        text = response.choices[0].message.content.strip()
        return text

    # Adapter pour rester compatible avec le reste du code (`llm.invoke(...).content`)
    def invoke(self, prompt: str) -> LLMResponse:
        text = self.generate(prompt)
        return LLMResponse(content=text)

