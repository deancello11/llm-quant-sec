#!/usr/bin/env bash
# Saves the exact version of every installed library to requirements.lock.txt, like writing down the
# recipe that worked. Run it after a successful setup so someone else (or future you) can rebuild
# the same environment later.
pip freeze > requirements.lock.txt && echo "wrote requirements.lock.txt. Commit it."
