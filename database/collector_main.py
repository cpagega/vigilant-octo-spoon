import ee
import os
from dotenv import load_dotenv
from collectors.FIRMS_collector import FIRMSCollector
from collectors.CONUS_collector import CONUSCollector
from collectors.CFSR_collector import CFSRCollector
from collectors.DaymetV4_collector import DaymetV4Collector

class Collector():
    def __init__(self):
        load_dotenv()
        ee.Authenticate()
        ee.Initialize(project=os.getenv("EE_PROJECT"))
        print("Google Earth Engine initialized successfully!")

    def collect_all(self):
        self.collect_FIRMS()
        self.collect_CONUS()
        self.collect_CFSR()
        self.collect_DaymetV4()
        self.collect_CPC_Precipitation

    def collect_FIRMS(self):
        fc =  FIRMSCollector()
        fc.collect_data()

    def collect_CONUS(self):
        cc = CONUSCollector()
        cc.collect_data()

    def collect_CFSR(self):
        cc = CFSRCollector()
        cc.collect_data()

    def collect_DaymetV4(self):
        d4 = DaymetV4Collector()
        d4.collect_data()

    def collect_CPC_Precipitation(self):
        pass



if (__name__ == "__main__"):
    collector = Collector()

    #We can add input to pick which set to collect
    collector.collect_DaymetV4()