#!/bin/sh
mkdir -p data
mkdir -p results
count=`ls -1 data/*.arff 2>/dev/null | wc -l`
if [ $count != 0 ]
then 
    echo true
else
    wget -nc http://promise.site.uottawa.ca/SERepository/datasets/kc2.arff &> /dev/null
    wget -nc http://promise.site.uottawa.ca/SERepository/datasets/pc1.arff &> /dev/null
    mv *.arff data
fi 
