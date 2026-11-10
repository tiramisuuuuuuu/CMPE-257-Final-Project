# Data formats we all agree on

- L: numpy bool array, shape (n_rows, n_literals); each column = one yes/no piece like "amount > 500"
- names: list of strings, the literal names, same order as L's columns
- y: numpy bool array, shape (n_rows,), True = fraud / default
- A rule: list of column indices, e.g. [3, 7] = column 3 AND column 7
- A rule set: set of rule indices into the rule pool; it fires if ANY rule fires
- A menu point: (precision, recall)
