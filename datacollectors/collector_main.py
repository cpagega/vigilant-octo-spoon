import ee
import os
from dotenv import load_dotenv
from collectors.FIRMS_collector import FIRMSCollector
#from collectors.EE_collector import EECollector
from collectors.test_collector import EECollector
"""
@author: Tyler
"""

class Collector():
    def __init__(self):
        load_dotenv()
        self.project_id = os.getenv("EE_PROJECT")
        ee.Authenticate()
        ee.Initialize(project=self.project_id)
        print("Google Earth Engine initialized successfully!")

    def collect_FIRMS(self):
        fc =  FIRMSCollector()
        fc.collect_data()

    def collect_EE(self):
        cfc = EECollector(self.project_id)
        cfc.collect_data()


2

if (__name__ == "__main__"):
    collector = Collector()
    functions = [collector.collect_FIRMS, collector.collect_EE]
    
    choice = int(input("" +
    "1) FIRMS\n" +
    "2) EE Data\n"
    "Enter Database to Collect: "
    ))
    functions[choice - 1]()