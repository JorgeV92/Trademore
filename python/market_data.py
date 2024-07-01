import json
from datetime import datetime, timedelta

import alpaca
from alpaca.data.live.stock import *
from alpaca.data.historical.stock import *
from alpaca.data.requests import *
from alpaca.data.timeframe import *
from alpaca.trading.client import *
from alpaca.trading.stream import *
from alpaca.trading.requests import *
from alpaca.trading.enums import *
from alpaca.common.exceptions import APIError

# keys required
API_KEY = "PKQ6ZYP9Q3HLDCVJB1TV"  
API_SECRET = "wjobG7qsUWMRfXlg6uGpWjmKWdqUf2MdTavc4OWA"



print("Compiled")