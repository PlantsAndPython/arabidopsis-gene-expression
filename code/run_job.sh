#!/bin/bash
# datatypes=("arabidopsis" "angiosperm");
datatypes=("arabidopsis");
#datatypes=("angiosperm");
#modeltypes=("SVC" "KNN" "MLP" "HGB" "RF");
metafactors=("TissueClean" "AboveBelow" "VegetativeRepro")
modeltypes=("KNN");
for d in ${datatypes[@]};
do
    for m in ${modeltypes[@]};
    do
        for f in ${metafactors[@]};
        do
            myjobname=$(echo $d)_$(echo $m)_$(echo $f);
            echo "Data Type: $d";
            echo "Model Type: $m";
            echo "Meta Factor: $f";
            echo "run_$(echo $d)_ml.py";
            echo "$myjobname.log";
            sbatch --job-name=$myjobname --output=$myjobname.SLURMout \
            --export=DATA=$d,MODEL=$m,FACTOR=$f run_ml_slurm_job.sb;
        done;
    done;
done;
