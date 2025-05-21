# Complex Network Effects on the Robustness of Graph Convolutional Networks
This repository contains code accompanying the paper "Complex Network Effects on the Robustness of Graph Convolutional Networks" in <i>Applied Network Science</i>, vol. 9, 2024.

Please note that this code runs code in the sequence outlined below, with intermediate results being saved to files as defined within the scripts.

## Hyperparameter Tuning
Hyperparameters were tuned using the following python scripts:

      baseline_hyper_parameters.py <dataset> <split_type> <defense> <attack> <nhid> <dropout> <lr> <weight_decay> <num_hops> <heads> <output_heads> <gamma> <beta1> <beta2> <svd_dim> <thres> <seed>

followed by

      update_defaults.py <dataset> <split_type> <defense> <attack>
where
* `<dataset>` is a string from the set {'cora', 'citeseer', 'polblogs', 'pubmed', 'er', 'ba', 'ws', 'lfr', 'config'}
* `<split_type>` is a string from the set {'random', 'degree', 'cover'}
* `<defense>` is a string from the set {'GCN', 'Jaccard', 'SVD', 'Cheb', 'median', 'GAT', 'RGCN', 'SGC'}
* `<attack>` is a string from the set {'Nettack', 'IGAttack', 'FGA', 'SGA'}
* `<nhid>` is a positive integer
* `<dropout>` is a value between 0 and 1
* `<lr>` is a floating-point number
* `<weight_decay>` is a number between 0 and 1
* `<num_hops>` is a postive integer
* `<heads>` is a positive integer
* `<output_heads>` is a positive integer
* `<gamma>` is a positive floating-point number
* `<beta1>` is a positive floating-point number
* `<beta2>` is a positive floating-point number
* `<svd_dim>` is a positive integer
* `<thres>` is a number between 0 and 1
* `<seed>` is a positive integer

## Target Node Selection
After tuning the hyperparameters, target nodes are selected by running the following script:

      create_targets.py <dataset> <split_type> <unlabeled_share> <defense> <attack> <seed> <feature_ind>
for real data, or

      create_targets.py <dataset> <split_type> <unlabeled_share> <defense> <attack> <feature_ind>
for synthetic data, where
* `<unlabeled_share>` is a number between 0 and 1 (usually set to 0.8)
* `<feature_ind>` is an integer in {0, 1, 2, 3}
and all other parameters are as above.

## Attack
Following target selection, attacks are run using

      run_attacks.py <dataset> <split_type> <direct_string> <target_string> <unlabeled_share> <defense> <attack> <feature_ind>
where
* `<direct_string>` is 0 or 1
* `<target_string>` is a string from the set {'struct', 'feat', 'both'}
and all other parameters are as above.
## Defense
Finally, defenses are run using

      run_defense.py <dataset> <split_type> <direct_string> <target_string> <unlabeled_share> <defense> <attack> <feature_ind>
      
## Acknowledgements and Disclaimers
This material is based upon work supported by the United States Air Force under
Air  Force  Contract  No.  FA8702-15-D-0001  and  the  Combat  Capabilities
Development Command Army Research Laboratory (under Cooperative Agreement Number
W911NF-13-2-0045).  Any  opinions,  findings,  conclusions  or  recommendations
expressed in this material are those of the authors and do not necessarily
reflect theviews of the United States Air Force or Army Research Laboratory.

Copyright (C) 2023
Benjamin A. Miller and Tina Eliassi-Rad

The software is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
Government rights in this work are defined by DFARS 252.227-7013 or DFARS
252.227-7014 as detailed above. Use of this work other than as specifically
authorized by the U.S. Government may violate any copyrights that exist in this
work
