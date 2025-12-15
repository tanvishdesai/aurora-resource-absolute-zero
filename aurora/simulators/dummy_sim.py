from .simulator import UnifiedResourceSimulator

class DummySimulator(UnifiedResourceSimulator):
    """
    A simplified simulator wrapper for initial testing and pipelining.
    Actually just uses the UnifiedResourceSimulator logic for now, 
    but allows us to mock outputs if needed later.
    """
    pass
