# initialize unit registry
from pint import UnitRegistry
# ureg is needed in simplig so the import can't be at the top of the file
ureg = UnitRegistry()
ureg.define("critical_density_800nm = 1.7401e27 / m^3 = n_c_800_ = n_c_800")

from .simplig import FieldMetaData, plot_field, OpenPMDDataLoader  # noqa: E402

__all__ = ['FieldMetaData', 'plot_field', 'OpenPMDDataLoader']
