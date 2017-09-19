#!/bin/bash


fast_tests() {
    make -C pysgmcmc test-notebooks
    py.test -v --doctest-modules --cov=pysgmcmc
}

all_tests() {
    # XXX Run slow tests here
    fast_tests
}


if [[ "$TRAVIS_EVENT_TYPE" -eq "cron" ]]; then
    # for the daily cron job, run *all* tests (and not just the fast-paced ones)
    all_tests
else
    fast_tests
fi
