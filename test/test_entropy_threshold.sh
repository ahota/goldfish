#!/bin/bash
nrounds=100
padlen=${#nrounds}
specmin=0
specmax=10
threshmin=1000
threshmax=5000
threshstep=100
datamin=16
datamax=128
datastep=16

runtest() {
    printf $'.PHONY: out\n'
    for (( spec = specmin; spec < specmax; spec++ )); do
        for (( thresh = threshmin; thresh <= threshmax; thresh += threshstep )); do
            printf $'out: out.%u.%u\n' $spec $thresh
            printf $'out.%u.%u:\n' $spec $thresh
            printf $'\tpython EntropyTest.py ../images/surface/supernova0000.0.%s.png -e %s -n %s --quiet; echo $$? > $@\n' \
                $spec $thresh $nrounds
        done
    done
}

make -j 48 -f <( runtest ) out

for (( spec = specmin; spec < specmax; spec++ )); do
    for (( thresh = threshmin; thresh < threshmax; thresh += threshstep )); do
        printf $'%*u ' $padlen $(cat out.$spec.$thresh)
    done
    printf $'\n'
done
