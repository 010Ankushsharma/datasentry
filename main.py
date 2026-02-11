import numpy as np
from datasentry import analyze

# Create dummy dataset
X = np.random.rand(100, 5)
X[0] = [100, 100, 100, 100, 100]  # force outlier

y = np.array([0]*80 + [1]*20)

# Analyze
report = analyze(X, y)

report.show()

# Try auto-fix
X_clean, y_clean = report.fixer.fix(X, y)

print("Before:", len(y))
print("After:", len(y_clean))
