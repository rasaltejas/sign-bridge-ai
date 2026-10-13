from typing import Any

__version__: str

class _HandsModule:
    HAND_CONNECTIONS: Any

    class Hands:
        def __init__(
            self,
            *,
            static_image_mode: bool = ...,
            max_num_hands: int = ...,
            min_detection_confidence: float = ...,
            min_tracking_confidence: float = ...,
            model_complexity: int = ...,
            **kwargs: Any,
        ) -> None: ...
        def process(self, image: Any, /) -> Any: ...
        def close(self) -> None: ...

class _DrawingUtilsModule:
    def draw_landmarks(
        self,
        image: Any,
        landmark_list: Any,
        connections: Any,
        /,
        **kwargs: Any,
    ) -> None: ...

class _SolutionsModule:
    hands: _HandsModule
    drawing_utils: _DrawingUtilsModule

solutions: _SolutionsModule

tasks: Any
