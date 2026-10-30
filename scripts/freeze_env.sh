#!/usr/bin/env bash
pip freeze > requirements.lock.txt && echo "wrote requirements.lock.txt. Commit it."
