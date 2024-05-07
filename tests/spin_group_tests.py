from rmatrix import Particle, ElasticChannel, CaptureChannel, SpinGroup
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

@pytest.fixture
def elastic(neutron,Ta181):
    J = 3
    pi = 1  # positivie parity
    ell = 0  # only s-waves are implemented right now
    radius = 0.2   # *10^(-12) cm 
    reduced_width_aplitudes = [106.78913185, 108.99600881]

    return ElasticChannel(neutron, Ta181,J,pi,ell,radius,reduced_width_aplitudes)

@pytest.fixture
def capture_ground(gamma,Ta182):
    J = 3
    pi = 1 
    ell = 1 
    radius = 0.2   
    reduced_width_aplitudes = [2.51487027e-06, 2.49890268e-06]
    excitation = 0  

    return CaptureChannel(gamma,Ta182,J,pi,ell,radius,reduced_width_aplitudes, excitation)


@pytest.fixture
def capture_first(gamma,Ta182):
    J = 3
    pi = 1  
    ell = 1 
    radius = 0.2   
    reduced_width_aplitudes = 0.8*np.array([2.51487027e-06, 2.49890268e-06])
    excitation = 5e5 

    return CaptureChannel(gamma,Ta182,J,pi,ell,radius,reduced_width_aplitudes, excitation)

@pytest.fixture
def res_energies():
    return  [1e6,1.1e6]

@pytest.fixture
def energy_grid():
    return np.linspace(0.9e6,1.2e6,1001)

def test_two_capture_channels(res_energies, energy_grid, elastic, capture_first, capture_ground):
    obj = SpinGroup(res_energies, elastic, [capture_ground,capture_first],energy_grid)

    exp_A = np.array([
        [ 9.96618054e-06+4.70344582e-07j, -1.72591797e-08+2.40032302e-07j],
        [-1.72591797e-08+2.40032302e-07j,  4.99119207e-06+1.22496374e-07j]
    ])

    assert np.allclose(obj.A_matrix[0,:,:],exp_A)

    assert np.isclose(obj.channels[0].cross_section[0], 0.33041047)