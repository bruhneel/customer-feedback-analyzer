# Held-out feedback evaluation

40 synthetic comments from 10 entirely held-out scenarios. Test metrics use a forced category to compare all models on the same rows.

| Model | Accuracy |
| --- | ---: |
| DistilGPT2 | 67.5% |
| Training majority | 20.0% |

## DistilGPT2

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| product_issue | 0.667 | 1.000 | 0.800 | 8 |
| staff_praise | 0.667 | 0.750 | 0.706 | 8 |
| pricing | 0.875 | 0.875 | 0.875 | 8 |
| delivery_or_stock | 1.000 | 0.500 | 0.667 | 8 |
| store_experience | 0.286 | 0.250 | 0.267 | 8 |

Confusion matrix, rows are actual and columns are predicted.

| Actual / predicted | product_issue | staff_praise | pricing | delivery_or_stock | store_experience |
| --- | ---: | ---: | ---: | ---: | ---: |
| product_issue | 8 | 0 | 0 | 0 | 0 |
| staff_praise | 0 | 6 | 0 | 0 | 2 |
| pricing | 1 | 0 | 7 | 0 | 0 |
| delivery_or_stock | 0 | 0 | 1 | 4 | 3 |
| store_experience | 3 | 3 | 0 | 0 | 2 |

## Training majority

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| product_issue | 0.200 | 1.000 | 0.333 | 8 |
| staff_praise | 0.000 | 0.000 | 0.000 | 8 |
| pricing | 0.000 | 0.000 | 0.000 | 8 |
| delivery_or_stock | 0.000 | 0.000 | 0.000 | 8 |
| store_experience | 0.000 | 0.000 | 0.000 | 8 |

Confusion matrix, rows are actual and columns are predicted.

| Actual / predicted | product_issue | staff_praise | pricing | delivery_or_stock | store_experience |
| --- | ---: | ---: | ---: | ---: | ---: |
| product_issue | 8 | 0 | 0 | 0 | 0 |
| staff_praise | 8 | 0 | 0 | 0 | 0 |
| pricing | 8 | 0 | 0 | 0 | 0 |
| delivery_or_stock | 8 | 0 | 0 | 0 | 0 |
| store_experience | 8 | 0 | 0 | 0 | 0 |

## Novel demonstration inputs

| Comment | Category | Score |
| --- | --- | ---: |
| The strap split the first time I put anything in the bag. | product_issue | 100.0% |
| The cashier was brilliant and went back to find the right size for me. | staff_praise | 100.0% |
| I was charged twice even though I only bought one bowl. | pricing | 90.7% |
| The tracking says delivered and there is no box anywhere. | delivery_or_stock | 100.0% |
| The shop was so crowded I couldn't get to the tills. | store_experience | 100.0% |
