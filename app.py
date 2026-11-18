# Entry point for hosts that expect app.py at the repository root.
# This delegates to the real app in src/app.py.
from src.app import main

if __name__ == "__main__":
    main()