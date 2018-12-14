#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Runs the continuumio/anaconda docker container, mapping repo-local folders
# 'notebooks' and 'data' as volumes in the container under '/opt/nola-traffic'
# and starting the Jupyter notebooks server
docker run -i -t -p 8888:8888 \
  -v "${DIR}/notebooks:/opt/nola-traffic/notebooks" \
  -v "${DIR}/data:/opt/nola-traffic/data" \
  continuumio/anaconda3:latest \
  /bin/bash -c "/opt/conda/bin/conda install jupyter geopandas -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/nola-traffic/notebooks --ip='*' --port=8888 --no-browser --allow-root"
