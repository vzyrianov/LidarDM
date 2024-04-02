from typing import Any, Dict, Optional, Tuple

__all__ = ["Visualizer"]


class Visualizer():
    def __init__(
            self,
            output_key: str,
    ):
        self.output_key = output_key

    def get_output_key(self) -> str:
        return self.output_key

    def supports_visualization(self, data: Dict[str, any]) -> bool:
        return False

    def generate_visualization(self, data: Dict[str, any]):
        return None
