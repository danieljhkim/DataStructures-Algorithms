import pandas as pd
from typing import *


# 1757. Recyclable and Low Fat Products
def find_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products[(products.low_fats == "Y" & products.recyclable == "Y")]
    return df["product_id"]


# 2877. Create a DataFrame from List
def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
    df = pd.DataFrame(data=student_data, columns=["student_id", "age"])
    return df


# 570. Managers with at Least 5 Direct Reports
def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
    df = employee.groupby(["managerId"]).count()
    df.show()
