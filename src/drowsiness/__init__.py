"""Fleet driver drowsiness and sleep detection.

Layers (each depends only on the ones above it):

    sources       -> frames from a webcam, a video file, or a recorded feature stream
    detection     -> face landmarks (MediaPipe), behind a swappable interface
    features      -> EAR, MAR, head pose, signal quality  (observations)
    analysis      -> calibration and temporal behaviour   (interpretation)
    safety        -> risk engine, state machine, alerts   (decision)
    events        -> structured logs and optional evidence
    presentation  -> on-screen overlay
"""

__version__ = "0.1.0"
