"""Vendored GELLO Dynamixel driver (the subset gello_node needs).

Copied from https://github.com/wuphilipp/gello_software at commit 204f53a
(MIT License, Copyright (c) 2023 Philipp Wu — see LICENSE in this directory):

    gello/dynamixel/driver.py -> driver.py
    gello/robots/dynamixel.py -> dynamixel.py
    gello/robots/robot.py     -> robot.py

Vendored rather than depended on because upstream gello isn't installable as
a regular package (``gello/robots`` has no ``__init__.py``, so only an editable
checkout works). Local edits: import paths, plus the
``EXTENDED_POSITION_CONTROL_MODE`` constant in driver.py. Requires the
``dynamixel-sdk`` package (``uv sync --extra gello``).
"""
