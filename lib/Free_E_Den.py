from symengine.lib.symengine_wrapper import solve
from symengine import sin, cos, sqrt, sympify, Symbol, Function, Eq
from dataclasses import dataclass, asdict, field

import math as m


class FreeEDen:
    def __init__(self, free_e: str):
        self.F = sympify(free_e)
        self._solve()

    def jac(self, args: tuple[str] = ('theta', 'phi')) -> str:
        F = self.F
        jac_list = []

        for arg in args:
            jac_list.append(F.diff(arg))
        return jac_list

    def _solve(self):
        F = self.F
        halb_res = sympify('gamma ** 2 / M ** 2') * (F.diff('theta', 2) * (F.diff('phi', 2) / (sin('theta')) ** 2
                    + cos('theta') / sin('theta') * F.diff('theta')) - (F.diff('theta', 'phi') / sin('theta')
                    - cos('theta') / sin('theta') * F.diff('phi') / sin('theta')) ** 2)

        eq_halb_res = Eq('omega ** 2', halb_res)
        b_res = solve(eq_halb_res, 'B')  # Solves the Baselgia approach
        omega_res = solve(eq_halb_res, 'omega')

        self.B_solutions = b_res
        self.B_res_args = b_res.atoms()

        self.Om_solutions = omega_res
        self.Om_res_args = omega_res.atoms()

if __name__ == '__main__':
    freeE = '-B*M*(sin(theta)*sin(thetaB)*cos(phi - phiB) + cos(theta)*cos(thetaB)) - K2p*sin(theta)**2*cos(phi ' \
            '- phiu)**2 - K4p*(cos(4*phi) + 3)*sin(theta)**4/8 - K4s*cos(theta)**4/2 - (-K2s + M**2*mu0/2)*' \
            'sin(theta)**2'

    F = FreeEDen(freeE)
    print(F.B_solutions.atoms())



