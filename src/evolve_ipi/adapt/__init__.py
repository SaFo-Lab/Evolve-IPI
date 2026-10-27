"""Self-evolution of the agent under test: RL on its weights, or skill evolution of its skill library."""


class AdaptAbort(Exception):
    """An adaptation run cannot go on; the stage or library it was building is discarded and rebuilt."""
