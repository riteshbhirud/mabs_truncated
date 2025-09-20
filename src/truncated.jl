"""
Truncated bosonic MPS algorithm using finite-dimensional local Hilbert spaces.
Represents each bosonic mode by truncating at a maximum occupation number.
"""
struct Truncated <: MabsAlg end