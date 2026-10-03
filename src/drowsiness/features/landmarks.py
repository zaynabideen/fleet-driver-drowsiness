"""MediaPipe Face Mesh landmark indices used by the feature extractors.

"Right"/"left" are the *driver's* right/left (the right eye appears on the
image's left in an un-mirrored camera).
"""

# EAR ordering: p1/p4 eye corners, p2/p3 upper lid, p5/p6 lower lid,
# with p2 vertically above p6 and p3 above p5.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)

RIGHT_EYE_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_CONTOUR = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)

# Inner-lip MAR: corners and three vertical pairs (upper, lower).
MOUTH_CORNERS = (78, 308)
MOUTH_VERTICAL_PAIRS = ((82, 87), (13, 14), (312, 317))

# Head-pose anchors.
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263
FOREHEAD = 10
CHIN = 152
NOSE_TIP = 1

NUM_LANDMARKS = 478
