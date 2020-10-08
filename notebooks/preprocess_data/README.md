# Replication of AI Clinician data preprocessing

The sequence of commands to run:
1. python AIClinician_Data_extract_MIMIC3_140219.py
2. matlab -nodisplay -nosplash -nojvm -r "run('AIClinician_sepsis3_def_160219.m'); exit;"
3. matlab -nodisplay -nosplash -nojvm -r "run('AIClinician_mimic3_dataset_160219.m'); exit;"
It is more convenient to start a MATLAB session and run the last 2 scripts in the same MATLAB  session. Note that these 3 files include configuration variables which are specific to setup (e.g., path names, the name of the database, etc)

All code in this folder is taken from the repository [matthieukomorowski/AI_Clinician](https://github.com/matthieukomorowski/AI_Clinician).
