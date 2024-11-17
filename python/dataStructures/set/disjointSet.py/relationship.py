from typing import List, Tuple, Optional
from collections import defaultdict


class DisjointSet:

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:  # for optimization, attach smaller tree to the bigger
            self.parent[rootY] = rootX

    #### problems ####

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """_summary_
        Args:
            accounts (List[List[str]]): ["name", "email1", "email2"]
        Returns:
            List[List[str]]: ["name", "email1", "email2"]
        """
        names = {}
        for acc in accounts:
            name = acc[0]
            emails = acc[1:]
            root_email = emails[0]
            for email in emails:
                if email not in self.parent:
                    self.parent[email] = email
                self.union(root_email, email)
                names[email] = name

        merged_emails = defaultdict(list)
        for email in self.parent:
            root = self.find(email)
            merged_emails[root].append(email)

        result = []
        for emails in merged_emails.values():
            name = names[emails[0]]
            result.append([name] + sorted(emails))
        return result


if __name__ == "__main__":
    s = DisjointSet()
    accounts = [
        ["John", "johnsmith@mail.com", "john00@mail.com"],
        ["John", "johnnybravo@mail.com"],
        ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
        ["Mary", "mary@mail.com"],
    ]
    result = s.accountsMerge(accounts)
    print("\nMerged accounts:")
    for account in result:
        print(account)
