# from pysgmcmc.sample_chains import Chain, MultiChain
from pysgmcmc.diagnostics.sample_chains import Chain, MultiChain
import unittest
from numpy.random import randint, rand

# XXX: Tests for chains stuff, check that they use cosmic ray whereever useful


"""
class MultiChainTests(object):
    def setup_method(self):
        pass
"""


class ChainConstructionTest(unittest.TestCase):
    def test_construction(self):
        self.n_parameters = 2
        self.n_samples = 2000

        self.sample_dimensions = (
            (2, 1),
            (3, 3)
        )

        def dummy_sampler():
            for _ in range(self.n_samples):
                cost = randint(2, 10)
                yield [
                    rand(*sample_dimension)
                    for sample_dimension in self.sample_dimensions
                ], cost

        """
        chain = [
            [[1.0, 2.0], [3.0, 4.0, 5.0]],
            [[2.0, 3.0], [6.0, 8.0, 10.0]],
        ]
        """
        parameter_names = [
            "dummy_{}".format(i) for i in range(len(self.sample_dimensions))
        ]

        c = Chain(
            sampler=dummy_sampler(), chain_length=self.n_samples,
            parameter_names=parameter_names
        )

        assert(c.max_samples == self.n_samples)
        print(c.mean())
        # print(c.samples)



class MultiChainConstructionTests(unittest.TestCase):
    def test_construction_with_parameter_names(self):
        self.n_parameters = 2
        self.n_chains = 2

        chain1 = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [3.0, 4.0]]
        ]

        chain2 = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [3.0, 4.0]]
        ]
        chains = [chain1, chain2,]

        parameter_names = ["x", "y"]

        mc = MultiChain(chain1, chain2, parameter_names=parameter_names)

        assert(mc.n_chains == self.n_chains)
        assert(mc.n_chains == self.n_parameters)
        assert(mc.parameter_names == parameter_names)

        for chain_index, chain in enumerate(chains):
            for sample in chain:
                for parameter_index, parameter in enumerate(sample):
                    # assert(parameter == mc.param_results[parameter_index][chain_index])
                    pass






if __name__ == "__main__":
    unittest.main()
