import ee
import os
from dotenv import load_dotenv
from collectors.FIRMS_collector import FIRMSCollector
from database.collectors.EE_collector import EECollector

class Collector():
    def __init__(self):
        load_dotenv()
        ee.Authenticate()
        ee.Initialize(project=os.getenv("EE_PROJECT"))
        print("Google Earth Engine initialized successfully!")

    def collect_all(self):
        self.collect_FIRMS()
        self.collect_EE()

    def collect_FIRMS(self):
        fc =  FIRMSCollector()
        fc.collect_data()

    def collect_EE(self):
        cfc = EECollector()
        cfc.collect_data()


if (__name__ == "__main__"):
    collector = Collector()
    functions = [collector.collect_FIRMS, collector.collect_EE, collector.collect_all]
    
    choice = int(input("" +
    "1) FIRMS\n" +
    "2) EE Data"
    "3) All\n" +
    "Enter Database to Collect: "
    ))
    functions[choice - 1]()