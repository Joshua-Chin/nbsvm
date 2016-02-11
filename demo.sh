#!/bin/bash

if [ ! -f aclImdb_v1.tar.gz ]; then
  wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
fi

if [ ! -f aclImdb ]; then
  tar -xf aclImdb_v1.tar.gz
fi

python3 demo.py
