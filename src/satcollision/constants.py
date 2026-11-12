"""Shared constants with no heavy imports.

Kept apart from twin.py (which pulls in sgp4, scipy and the fleet code) so
that processes which must never touch orbital data, above all the
federation server in federation_node.py, can import the signal and
reputation code without loading a propagator at all.
"""

# Industry-realistic default (see project definition, Sec. 3):
# ~1e-4 is the common threshold for standard maneuver planning,
# tighter (~1e-5) for high-value missions.
DEFAULT_PC_THRESHOLD = 1.0e-4
