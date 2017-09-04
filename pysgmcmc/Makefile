test:
	# XXX: Coverage
	py.test

tests: test

# XXX: Mutation testing with cosmic ray
mutation-tests:
	echo "Mutation tests not yet implemented"


# XXX: limit to fast-running tests
fast:
	echo "fast tests not yet implemented"

# XXX: Include more notebooks
nbval:
	py.test --nbval-lax ../notebooks/bayesian_neural_network/sinc_example.ipynb

doc:
	make -C doc html

.PHONY: doc
