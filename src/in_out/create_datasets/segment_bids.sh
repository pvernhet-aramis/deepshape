#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
DATADIR=${HOME}/'Data/MICCAI_dataset/4_segmentations/BIDS'
CAPSDIR=${HOME}/'Data/MICCAI_dataset/4_segmentations/segmented_BIDS'

clinica run t1-volume-tissue-segmentation ${DATADIR}/'1_training' ${CAPSDIR}$/'1_training' -wd ${CAPSDIR}$/'1_training_tmp' -np 40
clinica run t1-volume-tissue-segmentation ${DATADIR}/'2_validation' ${CAPSDIR}$/'2_validation' -wd ${CAPSDIR}$/'2_validation_tmp' -np 40
