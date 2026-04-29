class MotionGraph:
    def __init__(self, similarity_matrices: Dict[Tuple[str, str], np.ndarray]):
        self.similarity_matrices = similarity_matrices