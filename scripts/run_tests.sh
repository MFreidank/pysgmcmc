#!/bin/bash


fast_tests() {
    make -C pysgmcmc test-notebooks
    py.test -v --cov=pysgmcmc --doctest-modules --ignore=pysgmcmc/tests/bayesian_neural_network/test_bayesian_neural_network.py
}

all_tests() {
    py.test -v pysgmcmc/tests/bayesian_neural_network/test_bayesian_neural_network.py
    fast_tests
}

echo "TRAVIS: $TRAVIS_EVENT_TYPE"

if [[ "$TRAVIS_EVENT_TYPE" -eq "cron" ]]; then
    # for the daily cron job, run *all* tests (and not just the fast-paced ones)
    all_tests
else
    fast_tests
fi
