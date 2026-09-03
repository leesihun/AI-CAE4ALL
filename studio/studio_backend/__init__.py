"""AI-CAE4ALL Studio backend: a self-contained stdlib HTTP API for studio/.

This package only orchestrates and previews. It imports cae_suite for live
registry/spec/preflight metadata and subprocess-launches AI_CAE4ALL_main.py
and the method repos' own entrypoints; it never reimplements model logic.
"""
