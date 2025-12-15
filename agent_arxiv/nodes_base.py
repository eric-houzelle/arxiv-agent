from abc import ABC, abstractmethod

from .state import State


class Node(ABC):
    """Interface abstraite pour un nœud de workflow.

    Chaque nœud prend un `State` en entrée et retourne un `State` mis à jour.
    Cela permet d'avoir une structure extensible et testable pour les étapes
    du pipeline.
    """

    @abstractmethod
    def __call__(self, state: State) -> State:  # pragma: no cover - interface
        """Applique une étape de traitement au state."""
        raise NotImplementedError


class FunctionNode(Node):
    """Adaptateur simple pour transformer une fonction en nœud.

    Utile pour migrer progressivement les fonctions existantes (`search_arxiv`,
    `fetch_pdf_content`, etc.) vers une interface orientée objet.
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, state: State) -> State:
        return self._fn(state)

