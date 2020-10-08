#!/bin/bash

ENV_NAME='trajectory-inspection'

conda create -n ${ENV_NAME} python=3.7
conda install pandas -n ${ENV_NAME}
conda install scikit-learn -n ${ENV_NAME}
conda install jupyter -n ${ENV_NAME}
conda install numpy -n ${ENV_NAME}
conda install seaborn -n ${ENV_NAME}
conda install pickle -n ${ENV_NAME}
conda install matplotlib -n ${ENV_NAME}
conda install shelve -n ${ENV_NAME}
conda install tqdm -n ${ENV_NAME}
conda install logging -n ${ENV_NAME}
conda install scipy -n  ${ENV_NAME}
conda install time -n ${ENV_NAME}
conda install math -n ${ENV_NAME}
conda install sys -n ${ENV_NAME}
conda install setuptools -n ${ENV_NAME}
conda install collections -n ${ENV_NAME}
conda install ipywidgets -n ${ENV_NAME}
conda install tables -n ${ENV_NAME}
conda install psycopg2 -n ${ENV_NAME}
