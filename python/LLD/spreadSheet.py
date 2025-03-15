# 3484. Design Spreadsheet

"""_summary_

- 26 columns A - Z
- Each cell can hold an integer 0 - 10^5
"""


class Spreadsheet:

    def __init__(self, rows: int):
        self.sheet = {}
        self.R = rows
        self._build(rows)

    def _build(self, rows):
        for i in range(26):
            self.sheet[chr(i + ord("A"))] = [0] * rows

    def _get_key(self, cell):
        col = cell[0]
        row = int(cell[1:])
        return col, row - 1

    def setCell(self, cell: str, value: int) -> None:
        col, row = self._get_key(cell)
        self.sheet[col][row] = value

    def resetCell(self, cell: str) -> None:
        col, row = self._get_key(cell)
        self.sheet[col][row] = 0

    def getCellValue(self, cell):
        col, row = self._get_key(cell)
        return self.sheet[col][row]

    def getValue(self, formula: str) -> int:
        seg = formula.split("+")
        c1 = seg[0][1:]
        c2 = seg[1]
        val1 = int(c1) if not c1[0].isalpha() else self.getCellValue(c1)
        val2 = int(c2) if not c2[0].isalpha() else self.getCellValue(c2)
        return val1 + val2
