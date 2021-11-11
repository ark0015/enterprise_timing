#!/bin/bash

TOPDIR='/home/andrew/Documents/Research/nanograv/'
DATASET='12p5yr_tmp/'
TIMTYPE='*.tim'
PARDIR=$TOPDIR$DATASET'par/'
TIMDIR=$TOPDIR$DATASET'tim/'
NEWPARDIR=$TOPDIR$DATASET'T2refit_pars/'
NEWPARTYPE='.T2.par'
PARTYPE='*.par'
NEWPAR='new.par'

echo $DATASET
cd $TOPDIR$DATASET
for FILE in $TIMDIR$TIMTYPE
do
    TIMNAME=${FILE##*/}
    PSRNAME=${TIMNAME%%.tim*}
    echo $PSRNAME
    tempo2 -newpar -f $PARDIR$PSRNAME$PARTYPE $FILE
    echo "************"
    echo "FIT 1 Done."
    echo "************"
    tempo2 -newpar -f $PARDIR$PSRNAME$NEWPAR $FILE
    echo "************"
    echo "FIT 2 Done."
    echo "************"
    tempo2 -newpar -f $PARDIR$PSRNAME$NEWPAR $FILE
    echo "************"
    echo "FIT 3 Done."
    echo "************"
    cp $TOPDIR$DATASET$NEWPAR $NEWPARDIR$PSRNAME$NEWPARTYPE
    rm $TOPDIR$DATASET$NEWPAR
done