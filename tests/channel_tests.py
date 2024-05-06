from rmatrix import Particle, ElasticChannel
import pytest
import numpy as np

@pytest.fixture
def neutron():
    return Particle('n',1,1)

@pytest.fixture
def gamma():
    return Particle('g',0,0)

@pytest.fixture
def Ta181():
    return Particle('181Ta',181,73)

@pytest.fixture
def Ta182():
    return Particle('182Ta', 182, 73,Sn=6.8e6)

def test_elastic(neutron, Ta181):
    a_c= 0.2        # *10^(-12) cm
    neutron_widths = [106.78913185, 108.99600881]
    obj = ElasticChannel(neutron,Ta181,1,1,0,a_c,neutron_widths)

    assert obj.threshold == 0
    assert obj.ac == 0.2
    assert np.isclose(obj.calc_penetrability([1e-5])[0], 1.38191188e-06)