from RMQTool import Tools as RMTTools
import csv
from datetime import datetime


def run_live_call_model(indicatorEntity, signal):
    data = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
            round(indicatorEntity.tick_close, 3),
            signal,
            indicatorEntity.IE_assetsCode,
            indicatorEntity.IE_assetsName,
            indicatorEntity.IE_timeLevel]

    with open(RMTTools.read_config("RMQData", 'live_to_ts')
              + "A_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
