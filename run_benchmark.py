#!/usr/bin/env python
"""
Wrapper script for running JAM benchmarks.
This script maintains backward compatibility while using the new benchmark module.
"""

from benchmarks.cli import main

if __name__ == "__main__":
    main()