from rmatrix import Particle

def test_neutron():
    obj = Particle('n',1,0,0)
    assert obj.Sn == 0
    assert obj.label == 'n'
    assert obj.Z == 0
    assert obj.A == 1

def test_isotope():
    obj = Particle('182Ta',182,73,6.6e6)
    assert obj.Sn == 6.6e6