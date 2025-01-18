from collections import defaultdict


class UndergroundSystem:

    def __init__(self):
        self.checkin = {}
        self.counts = defaultdict(int)
        self.total = defaultdict(int)

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.checkin[id] = (t, stationName)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        start_time, station = self.checkin[id]
        time = t - start_time
        tag = f"{station}-{stationName}"
        self.counts[tag] += 1
        self.total[tag] += time

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        tag = f"{startStation}-{endStation}"
        return self.total[tag] / self.counts[tag]


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)
