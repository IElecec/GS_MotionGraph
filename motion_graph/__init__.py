__all__ = [
    "MotionGraph",
    "FrameRef",
    "GraphEdge",
    "Transition",
    "TransitionFrame",
    "build_transition_window",
    "build_transition_window_from_database",
    "build_transitions_from_matrices",
    "save_transition_window",
    "synthesize_transition_gaussian",
    "transition_from_dict",
]


def __getattr__(name: str):
    if name == "MotionGraph":
        from .motion_graph import MotionGraph

        return MotionGraph

    if name in {
        "FrameRef",
        "GraphEdge",
        "Transition",
        "TransitionFrame",
        "build_transition_window",
        "build_transition_window_from_database",
        "build_transitions_from_matrices",
        "save_transition_window",
        "synthesize_transition_gaussian",
        "transition_from_dict",
    }:
        from . import transition as _transition

        return getattr(_transition, name)

    raise AttributeError(name)
