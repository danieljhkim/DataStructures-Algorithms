from typing import *


class Dir:
    def __init__(self, name):
        self.dirs = {}
        self.files = {}
        self.name = name

    def list_all(self):
        return sorted(list(self.dirs.keys()) + list(self.files.keys()))


class File:
    def __init__(self, name, content=""):
        self.content = [content]
        self.name = name

    def add(self, data):
        self.content.append(data)

    def read(self):
        return "".join(self.content)


# 588. Design In-Memory File System
class FileSystem:

    def __init__(self):
        self.root = Dir("root")

    def ls(self, path: str) -> List[str]:
        paths = path.split("/")
        cur = self.root
        for p in paths[:-1]:
            if p == "":
                continue
            if p not in cur.dirs:
                return []
            cur = cur.dirs[p]
        last = paths[-1]
        if last == "":
            return cur.list_all()
        if last in cur.dirs:
            return cur.dirs[last].list_all()
        if last in cur.files:
            return [last]
        return []

    def mkdir(self, path: str) -> None:
        paths = path.split("/")
        cur = self.root
        for p in paths:
            if p == "":
                continue
            if p not in cur.dirs:
                cur.dirs[p] = Dir(p)
            cur = cur.dirs[p]

    def addContentToFile(self, filePath: str, content: str) -> None:
        paths = filePath.split("/")
        cur = self.root
        for p in paths[:-1]:
            if p == "":
                continue
            if p not in cur.dirs:
                return
            cur = cur.dirs[p]
        if paths[-1] in cur.files:
            cur.files[paths[-1]].add(content)
        else:
            cur.files[paths[-1]] = File(paths[-1], content)

    def readContentFromFile(self, filePath: str) -> str:
        paths = filePath.split("/")
        cur = self.root
        for p in paths[:-1]:
            if p == "":
                continue
            if p not in cur.dirs:
                return ""
            cur = cur.dirs[p]
        file_name = paths[-1]
        if file_name not in cur.files:
            return ""
        return cur.files[file_name].read()
