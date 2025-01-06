class SnapshotArray:

    def __init__(self, length: int):
        self.n = length
        self.arr = [[0] for _ in range(length)]
        self.id = 0

    def set(self, index: int, val: int) -> None:
        cur = self.arr[index]
        while len(cur) < self.id + 1:
            cur.append(cur[-1])
        self.arr[index][self.id] = val

    def snap(self) -> int:
        self.id += 1
        return self.id

    def get(self, index: int, snap_id: int) -> int:
        cur = self.arr[index]
        if snap_id < len(cur):
            return cur[snap_id]
        return cur[-1]
