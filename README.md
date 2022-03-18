# New Variations of Random Survival Forests and Applications to Age-Related Disease Data
These files have been used in the paper: T. Pomsuwan and A.A. Freitas. New Variations of Random Survival Forests and Applications to Age-Related Disease Data, to appear in the Proceedings of the 10th IEEE International Conference on Healthcare Informatics (ICHI 2022).

RSF_d0.py and sksurve_RSF.py are executable scripts to run different variants of RSF algorithm. 

This needs to be used in a combination with the scikit-survival module version 0.14.0 with the following modified scripts:

replacing tree.py in
.../sksurv/tree/tree.py by the corresponding file in the ‘modified_library’ folder of this site

replacing _criterion.c in
.../sksurv/tree/_criterion.c by the corresponding file in the ‘modified_library’ folder of this site

replacing forest.py in
.../sksurv/ensemble/forest.py by the corresponding file in the ‘modified_library’ folder of this site



Due to the license restrictions of the ELSA database, we can not make the ELSA datasets used in this work publicly available on this website. However, we could share our processed ELSA dataset with holders of the UKDS End User Licence (you will need to also accept the Additional condition for SN 5050). Please follow the procedure described at this site: https://www.elsa-project.ac.uk/accessing-elsa-data . You will need to read the description of the "Main ELSA Dataset" section, follow the "UK Data Service" link there and then register for access to the item: SN 5050. Once you have gained access to this item, you can send us an email and then we could send you the processed ELSA datasets used in this work (including the result of basic pre-processing/data cleaning to make the data suitable for a machine learning algorithm). The email address for requesting these processed datasets is tos_jack-4663@gmail.com.

Similarly, the SHARE dataset used in this work can not be made publicly available due to the non-transferrable right of use of the SHARE. However, an access to the SHARE data can be granted by the registration process at this site: http://share-project.org/data-access/ .
