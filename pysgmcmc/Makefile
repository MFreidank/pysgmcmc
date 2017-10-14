test:
	py.test
	make test-notebooks

# typo safety/convenience
tests: test

# Use "nbval" py.test plugin to evaluate all our notebooks and assert that
# they run without any errors
test-notebooks:
	for notebook in `find ../notebooks/* -name "*.ipynb"`; do \
	    py.test --nbval-lax $$notebook ; \
	done

# Run *full* test suite.
# This includes (amongst others) tests of our bnn with all supported samplers 
# on all synthetic hpolib functions.
# Note that this may take a long time (> 2h) and already happens once per day 
# on travis-ci.
full-test:
	cd .. && bash scripts/run_tests.sh --full

# typo safety/convenience
full-tests: full-test

# Run mutation testing. Note that this may take a long time (> 2h).
# Will construct mutations of our code and check if the tests detect the
# mutations. This allows for some validation of our test suite 
# to see how well it tracks our codebase.
mutation-tests:
	cd .. && cosmic-ray run --timeout 20 pysgmcmc tests | tee mutation_test_results.txt

doc:
	make -C ../docs html

.PHONY: doc
