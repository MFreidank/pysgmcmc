test:
	py.test -v --cov=pysgmcmc
	py.test --nbval-lax ../notebooks/api_quickstart.ipynb

tests: test

# XXX: Mutation testing with cosmic ray
mutation-tests:
	echo "Mutation tests not yet implemented"


# XXX: limit to fast-running tests
fast:
	echo "fast tests not yet implemented"

doc:
	make -C ../docs html

.PHONY: doc
