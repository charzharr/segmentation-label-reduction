#!/bin/bash

EXP_NAME = "hello_test"

#$ -N annotation_reduction
#$ -o "/afs/crc.nd.edu/user/y/yzhang46/_ANTS/${EXP_NAME}.txt"

#$ -M yzhang46@nd.edu
#$ -m abe                # Send mail when job begins, ends and aborts

#$ -pe smp 4             
#$ -q long          

#$ -q gpu@@csecri-p100   # gpu, gpu@@csecri-p100, gpu@@csecri-titanxp 
#$ -l gpu_card=1       


if [ "$USER" == "yzhang46" ]; then

        # Env and Requirements Setup
        cd /afs/crc.nd.edu/user/y/yzhang46/_ANNO3D

        module load python 
        module load pytorch

        echo -e "\n>>> Installing Python requirements\n"
        pip3 install --user -r requirements.txt
        echo -e "\n>>> Done installing Python requirements\n"

        echo -e "\n>>> Logging in to W&B\n"
        . ./scripts/wandb_login.sh
        echo -e "\n>>> Done logging in to W&B\n"

        cd ~/_ANNO/src

else

        cd "/Users/charzhar/Desktop/[Project] Annotation Reduction/_ANNO/"

        echo -e "\n>>> Installing Python requirements\n"
        pip3 install -r requirements.txt
        echo -e "\n>>> Done installing Python requirements\n"
        
        cd "/Users/charzhar/Desktop/[Project] Annotation Reduction/_ANNO/src"

fi


CUDA_VISIBLE_DEVICES=0,1,2,3
if [ -z ${SGE_HGR_gpu_card+x} ]; then 
        SGE_HGR_gpu_card=-1
fi
echo -e "Assigned GPU(s): ${SGE_HGR_gpu_card}\n"
echo -e "Starting Experiment =)"
echo -e "=-=-=-=-=-=-=-=-=-=-=-=-=\n"


python3 -u train.py \
        --config "./experiments/baseline2d/config.yaml" \
        --checkpoint "" \
        --user $USER  \
        --gpu $SGE_HGR_gpu_card

