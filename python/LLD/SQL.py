from typing import *

# 2408. Design SQL


class SQL:

    INVALID = "<null>"

    def __init__(self, names: List[str], columns: List[int]):
        self.database = {}
        for i, name in enumerate(names):
            new_table = Table(name, columns[i])
            self.database[name] = new_table

    def ins(self, name: str, row: List[str]) -> bool:
        if name not in self.database:
            return False
        return self.database[name].ins(row)

    def rmv(self, name: str, rowId: int) -> None:
        if name not in self.database:
            return
        self.database[name].rmv(rowId)

    def sel(self, name: str, rowId: int, columnId: int) -> str:
        if name not in self.database:
            return self.INVALID
        return self.database[name].sel(rowId, columnId)

    def exp(self, name: str) -> List[str]:
        if name not in self.database:
            return []
        return self.database[name].exp()


class Table:

    INVALID = "<null>"

    def __init__(self, name: str, column: int):
        self.name = name
        self.idx = 1
        self.rows = {}
        self.N = column

    def ins(self, row: List[str]) -> bool:
        if self.N != len(row):
            return False
        self.rows[self.idx] = row
        self.idx += 1
        return True

    def rmv(self, rowId: int) -> None:
        if rowId in self.rows:
            del self.rows[rowId]

    def sel(self, rowId: int, columnId: int) -> str:
        if rowId not in self.rows or not (0 < columnId <= self.N):
            return self.INVALID
        row = self.rows[rowId]
        return row[columnId - 1]

    def exp(self) -> List[str]:
        records = []
        for k, v in self.rows.items():
            records.append(f"{k},{','.join(v)}")
        return records
