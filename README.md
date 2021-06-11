# new_variants_of_RSF
used in a paper under review as an anonymous submission for a conference.

RSF_d0.py and sksurve_RSF.py are executable scripts to run different variants of RSF algorithm. 

This needs to be used in a combination with the scikit-survival module version 0.14.0 with the following modified scripts:

replacing tree.py in
.../sksurv/tree/tree.py

replacing _criterion.c in
.../sksurv/tree/_criterion.c

replacing forest.py in
.../sksurv/ensemble/forest.py
