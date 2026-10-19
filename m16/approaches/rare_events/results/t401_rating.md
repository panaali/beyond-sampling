# T401 safe-use rating table

r(t) = estimated error chance under the condition; check = conditional-law MC rate (hits / n);
plain MC needed = inputs plain sampling needs to see ~10 such errors.

## M_16,10 (the target): P(answers with the 3rd largest | g23 >= t)

| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.244 | 0.000759 | 0.000525 (1049 / 2e+06) | 1.45 | 0.60 | unrated | 5.4e+04 |
| 1.0 | 0.0444 | 0.000214 | 0.000127 (1016 / 8e+06) | 1.69 | 0.81 | unrated | 1.1e+06 |
| 1.5 | 0.00595 | 1.69e-05 | 3.71e-05 (1019 / 3e+07) | 0.46 | 0.93 | unrated | 9.9e+07 |
| 2.0 | 0.000578 | 2.44e-06 | 1.24e-05 (1008 / 8e+07) | 0.20 | 0.92 | unrated | 7.1e+09 |
| 2.5 | 4.03e-05 | 1.78e-08 | 3.61e-06 (361 / 1e+08) | 0.00 | 0.84 | unrated | 1.4e+13 |
| 3.0 | 2e-06 | 4.24e-10 | 9.3e-07 (93 / 1e+08) | 0.00 | 0.81 | unrated | 1.2e+16 |
| 3.5 | 7.04e-08 | 1.11e-14 | < 30 hits in 1e+08 | - | 0.92 | unrated | 1.3e+22 |

## M_16,10 (the target): P(any wrong answer | g12, g23 >= t (unconfirmed))

| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.0912 | 0.00106 | 0.00106 (1059 / 1e+06) | 1.00 | - | rated (validated) | 1e+05 |
| 0.75 | 0.0208 | 0.000254 | 0.000262 (1046 / 4e+06) | 0.97 | - | rated (validated) | 1.9e+06 |
| 1.0 | 0.00388 | 5.43e-05 | 5.43e-05 (1005 / 2e+07) | 1.00 | - | rated (validated) | 4.7e+07 |
| 1.25 | 0.00059 | 1.22e-05 | 1.26e-05 (1005 / 8e+07) | 0.97 | - | rated (validated) | 1.4e+09 |
| 1.5 | 7.26e-05 | 2.07e-06 | 3.38e-06 (338 / 1e+08) | 0.61 | - | rated (validated) | 6.6e+10 |

## retrain dev3_s529: P(answers with the 3rd largest | g23 >= t)

| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.244 | 0.00438 | 0.00395 (1975 / 5e+05) | 1.11 | 0.30 | rated (validated) | 9.4e+03 |
| 1.0 | 0.0444 | 0.00319 | 0.00302 (1512 / 5e+05) | 1.06 | 0.30 | rated (validated) | 7.1e+04 |
| 1.5 | 0.00595 | 0.00272 | 0.00247 (1233 / 5e+05) | 1.10 | 0.32 | rated (validated) | 6.2e+05 |
| 2.0 | 0.000578 | 0.00228 | 0.00188 (1883 / 1e+06) | 1.21 | 0.38 | rated (validated) | 7.6e+06 |
| 2.5 | 4.03e-05 | 0.00152 | 0.0014 (1400 / 1e+06) | 1.08 | 0.46 | rated (validated) | 1.6e+08 |
| 3.0 | 2e-06 | 0.000923 | 0.0011 (1100 / 1e+06) | 0.84 | 0.58 | unrated | 5.4e+09 |
| 3.5 | 7.04e-08 | 0.000574 | 0.000861 (1292 / 2e+06) | 0.67 | 0.66 | unrated | 2.5e+11 |

## retrain dev3_s529: P(any wrong answer | g12, g23 >= t (unconfirmed))

| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.0912 | 0.00373 | 0.00322 (1612 / 5e+05) | 1.16 | - | rated (validated) | 2.9e+04 |
| 0.75 | 0.0208 | 0.00226 | 0.00198 (1975 / 1e+06) | 1.14 | - | rated (validated) | 2.1e+05 |
| 1.0 | 0.00388 | 0.00174 | 0.00145 (1448 / 1e+06) | 1.20 | - | rated (validated) | 1.5e+06 |
| 1.25 | 0.00059 | 0.0014 | 0.00107 (1066 / 1e+06) | 1.32 | - | rated (validated) | 1.2e+07 |
| 1.5 | 7.26e-05 | 0.00104 | 0.000745 (1117 / 2e+06) | 1.39 | - | rated (validated) | 1.3e+08 |

## retrain dev3_s531: P(answers with the 3rd largest | g23 >= t)

| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.244 | 0.00886 | 0.00944 (4722 / 5e+05) | 0.94 | 0.48 | rated (validated) | 4.6e+03 |
| 1.0 | 0.0444 | 0.00726 | 0.00679 (3393 / 5e+05) | 1.07 | 0.45 | rated (validated) | 3.1e+04 |
| 1.5 | 0.00595 | 0.00586 | 0.00483 (2415 / 5e+05) | 1.21 | 0.44 | rated (validated) | 2.9e+05 |
| 2.0 | 0.000578 | 0.00373 | 0.00311 (1553 / 5e+05) | 1.20 | 0.44 | rated (validated) | 4.6e+06 |
| 2.5 | 4.03e-05 | 0.00237 | 0.00224 (1121 / 5e+05) | 1.06 | 0.42 | rated (validated) | 1e+08 |
| 3.0 | 2e-06 | 0.00203 | 0.00177 (1765 / 1e+06) | 1.15 | 0.48 | rated (validated) | 2.5e+09 |
| 3.5 | 7.04e-08 | 0.00188 | 0.00136 (1358 / 1e+06) | 1.38 | 0.47 | rated (validated) | 7.6e+10 |

## retrain dev3_s531: P(any wrong answer | g12, g23 >= t (unconfirmed))

| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.0912 | 0.00255 | 0.00408 (2042 / 5e+05) | 0.63 | - | rated (validated) | 4.3e+04 |
| 0.75 | 0.0208 | 0.00156 | 0.00235 (1175 / 5e+05) | 0.66 | - | rated (validated) | 3.1e+05 |
| 1.0 | 0.00388 | 0.000914 | 0.00117 (1170 / 1e+06) | 0.78 | - | rated (validated) | 2.8e+06 |
| 1.25 | 0.00059 | 0.000376 | 0.000539 (1077 / 2e+06) | 0.70 | - | rated (validated) | 4.5e+07 |
| 1.5 | 7.26e-05 | 7.71e-05 | 0.000217 (1084 / 5e+06) | 0.36 | - | rated (validated) | 1.8e+09 |

Cost (FPE, deductive): orig 3.78e+06, dev3_s529 3.95e+06, dev3_s531 3.15e+06
