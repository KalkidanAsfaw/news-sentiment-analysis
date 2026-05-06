def test_environment():
    """Verify the Python environment is set up correctly."""
    import pandas as pd
    import numpy as np
    assert pd.__version__ >= "1.5.0"
    assert np.__version__ >= "1.23.0"
