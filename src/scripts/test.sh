
if [ "$USER" == "yzhang46" ]; then

        module load python 
        module load pytorch

        cd ~/_ANNO3D/src
else
        cd "/Users/charzhar/Desktop/[Project] Annotation Reduction/ANNO_3D/src"
fi

python -m pytest tests