"""Run the live driver monitor on the webcam:

    python -m drowsiness                 # webcam 0
    python -m drowsiness --source 1      # another camera
    python -m drowsiness --help
"""

from drowsiness.cli import demo_main

if __name__ == "__main__":
    raise SystemExit(demo_main())
