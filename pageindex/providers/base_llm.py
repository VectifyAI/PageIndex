
from abc import ABC, abstractmethod

class BaseLLM(ABC):

    @abstractmethod
    def generate(self, model: str, messages: list, **kwargs):
        pass

    @abstractmethod
    async def agenerate(self, model: str, messages: list, **kwargs):
        pass