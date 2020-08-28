import unittest
from grid.loads import Loads


class LoadTests(unittest.TestCase):
    cl = Loads(path="/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/timeseries.json",
               conf='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json')
    print(cl.get_unique(name="A1", col="PROFILEID", loads="LOADVALUE_list", timesteps=4))
    print(cl.get_all(col="PROFILEID", loads="LOADVALUE_list"))