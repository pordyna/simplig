# initialize unit registry
from pint import UnitRegistry
ureg = UnitRegistry()
ureg.define("critical_density_800nm = 1.7401e27 / m^3 = n_c_800_ = n_c_800")
from .simplig import *

