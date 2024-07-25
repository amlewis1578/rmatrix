from rmatrix import Particle, ElasticChannel, CaptureChannel, SpinGroup
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def neutron():
    return Particle('n',1,1)

@pytest.fixture
def gamma():
    return Particle('g',0,0)

@pytest.fixture
def target():
    return Particle("20Ne",20,10)

@pytest.fixture
def compound():
    return Particle("20Ne", 20,10, Sn=6.6e6)

@pytest.fixture
def sammy_answers_bb():
    file_location = Path(__file__).parent / "files" / "rbb.lst"
    return np.loadtxt(file_location)

def test_bb_answers(neutron,gamma,target,compound,sammy_answers_bb):
    energies = np.array([1e6,1.1e6])
    energy_grid = sammy_answers_bb[:,0]*1e3 # convert to eV

    J = 0.5
    pi = 1  # positive parity
    ell = 0  #  s-wave
    radius = 0.532   # *10^(-12) cm 

    elastic_widths = np.array([1e7,1.1e7])/1e3   # widths from sammy in meV
    penetrabilities = np.array([1.11567655, 1.17013143])
    elastic_reduced_width_amplitudes = np.sqrt(elastic_widths/(2*penetrabilities))
    elastic = ElasticChannel(neutron,target,J,pi,ell,radius,elastic_reduced_width_amplitudes)

    capture_widths = np.array([1e3,1.1e3])/1e3   # widths from sammy in meV
    penetrabilities = np.array([0.20489882, 0.20759486])
    capture_reduced_width_amplitudes = np.sqrt(capture_widths/(2*penetrabilities))
    capture = CaptureChannel(gamma,compound,J,pi,ell,radius,capture_reduced_width_amplitudes,0)

    sg = SpinGroup(energies,elastic,[capture],energy_grid)

    assert np.allclose(sg.channels[1].cross_section,sammy_answers_bb[:,3],atol=5e-6)

def test_partial_widths_elastic(neutron,target):
    res_energies = np.array([1e6,1.1e6])

    J = 0.5
    pi = 1  # positive parity
    ell = 0  #  s-wave
    radius = 0.532   # *10^(-12) cm 

    elastic_widths = np.array([1e7,1.1e7])/1e3   # widths from sammy [in meV] to eV
    penetrabilities = np.array([1.11567655, 1.17013143])
    elastic_reduced_width_amplitudes = np.sqrt(elastic_widths/(2*penetrabilities))
    elastic = ElasticChannel(neutron,target,J,pi,ell,radius,partial_widths=elastic_widths, resonance_energies=res_energies)

    assert np.allclose(elastic.reduced_width_amplitudes, elastic_reduced_width_amplitudes)

    
def test_partial_widths_capture(gamma,compound):
    res_energies = np.array([1e6,1.1e6])

    J = 0.5
    pi = 1  # positive parity
    ell = 0  #  s-wave
    radius = 0.532   # *10^(-12) cm 

    capture_widths = np.array([1e3,1.1e3])/1e3   # widths from sammy in meV
    penetrabilities = np.array([0.20489882, 0.20759486])
    capture_reduced_width_amplitudes = np.sqrt(capture_widths/(2*penetrabilities))
    
    capture = CaptureChannel(gamma,compound,J,pi,ell,radius,partial_widths=capture_widths, resonance_energies=res_energies, excitation=0)
    
    assert np.allclose(capture.reduced_width_amplitudes, capture_reduced_width_amplitudes)