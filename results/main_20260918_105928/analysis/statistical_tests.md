# Statistische Tests (McNemar, gepaarter Vergleich)

Explorative Auswertung: Wiederholungen derselben Aufgabe sind nicht zwingend unabhaengig. Die p-Werte sind nicht fuer Mehrfachtests korrigiert und duerfen allein keine bestaetigenden Signifikanzbehauptungen begruenden.

Getestet wird, ob sich zwei Bedingungen bei denselben Aufgaben signifikant unterscheiden. p < 0.05 gilt ueblicherweise als signifikant, sollte bei kleinen Stichproben mit Vorsicht interpretiert werden.


## Quantisierung: Q4_K_M

| Vergleich | n Paare | Accuracy A | Accuracy B | nur A richtig | nur B richtig | p-Wert | signifikant (p<0.05) |
|---|---|---|---|---|---|---|---|
| H1: Budget 256 vs. 1024 (C, kein Zusatzprompt) | 120 | 38.3% | 70.0% | 4 | 42 | 0.0000 | JA |
| H1: Budget 256 vs. 1024 (D, generische Intervention) | 120 | 40.0% | 67.5% | 8 | 41 | 0.0000 | JA |
| H1: Budget 256 vs. 1024 (E, finanzspezifische Intervention) | 120 | 40.0% | 65.8% | 5 | 36 | 0.0000 | JA |
| H3: D vs. E bei Budget 256 | 120 | 40.0% | 40.0% | 5 | 5 | 1.0000 | nein |
| H3: D vs. E bei Budget 1024 | 120 | 67.5% | 65.8% | 4 | 2 | 0.6875 | nein |

## Quantisierung: Q8_0

| Vergleich | n Paare | Accuracy A | Accuracy B | nur A richtig | nur B richtig | p-Wert | signifikant (p<0.05) |
|---|---|---|---|---|---|---|---|
| H1: Budget 256 vs. 1024 (C, kein Zusatzprompt) | 120 | 39.2% | 65.8% | 7 | 39 | 0.0000 | JA |
| H1: Budget 256 vs. 1024 (D, generische Intervention) | 120 | 31.7% | 65.0% | 5 | 45 | 0.0000 | JA |
| H1: Budget 256 vs. 1024 (E, finanzspezifische Intervention) | 120 | 40.0% | 62.5% | 9 | 36 | 0.0001 | JA |
| H3: D vs. E bei Budget 256 | 120 | 31.7% | 40.0% | 2 | 12 | 0.0129 | JA |
| H3: D vs. E bei Budget 1024 | 120 | 65.0% | 62.5% | 6 | 3 | 0.5078 | nein |