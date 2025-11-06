import ee
import os
from collectors.FIRMS_builder import FIRMS_Builder

class Collector():
    def __init__(self):
        ee.Authenticate()
        print(f"Project Name: {os.getenv('EE_PROJECT')}")
        ee.Initialize(project=os.getenv("EE_PROJECT"))
        print("Google Earth Engine initialized successfully!")

    def collect_all(self):
        self.collect_FIRMS()
        self.collect_CRONUS()
        self.collect_CFSR()
        self.collect_DaymetV4()
        self.collect_CPC_Precipitation

    def collect_FIRMS(self):
        fb =  FIRMS_Builder()
        fb.collect_data()

    def collect_CRONUS(self):
        pass

    def collect_CFSR(self):
        pass

    def collect_DaymetV4(self):
        pass

    def collect_CPC_Precipitation(self):
        pass



if (__name__ == "__main__"):
    collector = Collector()

    #We can add input to pick which set to collect
    collector.collect_FIRMS()