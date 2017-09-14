test:
	py.test -v --cov=pysgmcmc
	make test-notebooks

tests: test

# XXX: Mutation testing with cosmic ray
mutation-tests:
	echo "Mutation tests not yet implemented"

# Use "nbval" py.test plugin to evaluate all notebooks and assert that
# they run without any errors
test-notebooks:
	for notebook in `find ../notebooks/* -name "*.ipynb"`; do \
	    py.test --nbval-lax $$notebook ; \
	done


# XXX: limit to fast-running tests
fast:
	echo "fast tests not yet implemented"

doc:
	make -C ../docs html

.PHONY: doc
