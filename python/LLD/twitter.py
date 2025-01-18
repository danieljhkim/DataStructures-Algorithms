from typing import *
from collections import defaultdict, deque
import heapq


class Twitter:

    def __init__(self):
        self.users = defaultdict(set)
        self.tweets = defaultdict(deque)
        self.idx = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.idx -= 1
        entry = (self.idx, tweetId)
        self.tweets[userId].append(entry)
        if len(self.tweets[userId]) > 10:
            self.tweets[userId].popleft()

    def getNewsFeed(self, userId: int) -> List[int]:
        heap = []
        feed = []
        heap.extend(self.tweets[userId])
        for us in self.users[userId]:
            heap.extend(self.tweets[us])
        heapq.heapify(heap)
        while heap and len(feed) < 10:
            out = heapq.heappop(heap)
            feed.append(out[1])
        return feed

    def follow(self, followerId: int, followeeId: int) -> None:
        self.users[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.users[followerId]:
            self.users[followerId].remove(followeeId)
