#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
DATADIR=${HOME}/'Data/MICCAI_dataset/4_segmentations/BIDS'
CAPSDIR=${HOME}/'Data/MICCAI_dataset/4_segmentations/segmented_BIDS'

# Data must be recentered previously using clinica command suggestion ...
clinica run t1-volume-tissue-segmentation ${DATADIR}/'1_training_centered' ${CAPSDIR}/'1_training_centered' -wd ${CAPSDIR}/'1_training_centered_tmp' -np 65
clinica run t1-volume-tissue-segmentation ${DATADIR}/'2_validation_centered' ${CAPSDIR}/'2_validation_centered' -wd ${CAPSDIR}/'2_validation_centered_tmp' -np 65
