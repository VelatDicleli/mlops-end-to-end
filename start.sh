#!/bin/bash

uvicorn app:app --host 0.0.0.0 --port 8000 &


python ui.py
