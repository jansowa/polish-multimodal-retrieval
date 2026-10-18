from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..config import Settings
from ..models import TranslationUnit


class DatasetAdapter(ABC):
    @abstractmethod
    def iter_units(self, settings: Settings, with_image: bool) -> Iterator[TranslationUnit]:
        raise NotImplementedError
