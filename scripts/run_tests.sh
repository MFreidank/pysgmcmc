#!/bin/bash


fast_tests() {
    make -C pysgmcmc test-notebooks
    py.test -v --doctest-modules --cov=pysgmcmc --ignore=pysgmcmc/tests/bayesian_neural_network/test_bayesian_neural_network.py
}

all_tests() {
    py.test -v pysgmcmc/tests/bayesian_neural_network/test_bayesian_neural_network.py
    fast_tests
}


if [[ "$TRAVIS_EVENT_TYPE" == "cron" ]]; then
    # for the daily cron job, run *all* tests (and not just the fast-paced ones)
    echo "Running all tests!"
    all_tests
else
    echo "Running only fast tests, sampler tests are only run via cronjobs."
    fast_tests
fi
