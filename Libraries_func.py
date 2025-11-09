try:
    import streamlit
    import pandas
    import numpy
    import matplotlib
    import yfinance
    import sklearn
    import tensorflow
    print("\n\n All libraries imported successfully!\n\n")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
