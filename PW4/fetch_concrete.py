import os
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)
X = concrete.data.features
y = concrete.data.targets

print(concrete.metadata)
print(concrete.variables)

X = X.copy()
X.columns = ["cement", "slag", "ash", "water", "superplastic",
             "coarse_agg", "fine_agg", "age"]
y = y.iloc[:, 0].rename("strength")

os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)