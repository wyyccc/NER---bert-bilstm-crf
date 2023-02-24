from pprint import pprint
from paddlenlp import Taskflow


schema = ['时间', '人物', '赛事']
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("12月6日上午卡塔尔世界杯中国队以满分100分获得金牌！"))