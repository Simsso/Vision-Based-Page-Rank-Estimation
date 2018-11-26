#!/bin/bash

Xvfb :99 &
export DISPLAY=:99

/opt/apt/datacrawler/datacrawler --no-sandbox --headless
