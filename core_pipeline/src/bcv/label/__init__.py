"""Browser-based labelling GUI (served over an SSH tunnel) for growing our labelled data.

A generic FastAPI + HTML5-canvas shell (frame server + label panel + scrubber) with
pluggable modes. First mode: the bounding-box placer (model-assisted red/blue box
labelling -> canonical fighter_bboxes.json). See ``app.py`` for routes, ``boxes.py`` for
the keyframe model, ``__main__.py`` for the ``bcv-label`` CLI.
"""
