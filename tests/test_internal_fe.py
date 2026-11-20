import json,pathlib,unittest
import numpy as np
from fasttwfepy._fe import FEAbsorber,ConvergenceError
class InternalFETests(unittest.TestCase):
    def test_common_projection_fixtures(self):
        root=pathlib.Path(__file__).parent/'fe_fixtures'
        for f in json.loads((root/'manifest.json').read_text()):
            with self.subTest(fixture=f['name']):
                data=np.genfromtxt(root/(f['name']+'.csv'),delimiter=',',names=True)
                a=FEAbsorber(*[data['g'+str(i+1)] for i in range(f['k'])],weights=data['w'])
                values=np.column_stack([data['v'+str(i+1)] for i in range(f['p'])])
                if not f['expected_converged']:
                    with self.assertRaises(ConvergenceError):a.residualize(values,max_iter=f['max_iter'])
                    continue
                actual=a.residualize(values);expected=np.loadtxt(root/(f['name']+'_oracle.csv'),delimiter=',',skiprows=1,ndmin=2)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
                np.testing.assert_allclose(a.residualize(actual),actual,rtol=0,atol=1e-9)
