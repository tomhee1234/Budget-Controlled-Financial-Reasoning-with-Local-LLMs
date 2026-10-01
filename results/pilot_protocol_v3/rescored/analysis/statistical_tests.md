# Statistische Tests (McNemar, gepaarter Vergleich)

Explorative Auswertung: Wiederholungen derselben Aufgabe sind nicht zwingend unabhaengig. Die p-Werte sind nicht fuer Mehrfachtests korrigiert und duerfen allein keine bestaetigenden Signifikanzbehauptungen begruenden.

Getestet wird, ob sich zwei Bedingungen bei denselben Aufgaben signifikant unterscheiden. p < 0.05 gilt ueblicherweise als signifikant, sollte bei kleinen Stichproben mit Vorsicht interpretiert werden.


## Quantisierung: Q4_K_M

| Vergleich | n Paare | Accuracy A | Accuracy B | nur A richtig | nur B richtig | p-Wert | signifikant (p<0.05) |
|---|---|---|---|---|---|---|---|
| H1: Budget 256 vs. 1024 (C, kein Zusatzprompt) | 4 | 0.0% | 50.0% | 0 | 2 | 0.5000 | nein |
| H1: Budget 256 vs. 1024 (D, generische Intervention) | 4 | 0.0% | 50.0% | 0 | 2 | 0.5000 | nein |
| H1: Budget 256 vs. 1024 (E, finanzspezifische Intervention) | 4 | 25.0% | 50.0% | 1 | 2 | 1.0000 | nein |
| H3: D vs. E bei Budget 256 | 4 | 0.0% | 25.0% | 0 | 1 | 1.0000 | nein |
| H3: D vs. E bei Budget 1024 | 4 | 50.0% | 50.0% | 0 | 0 | 1.0000 | nein |

## Quantisierung: Q8_0

| Vergleich | n Paare | Accuracy A | Accuracy B | nur A richtig | nur B richtig | p-Wert | signifikant (p<0.05) |
|---|---|---|---|---|---|---|---|
| H1: Budget 256 vs. 1024 (C, kein Zusatzprompt) | 4 | 25.0% | 50.0% | 0 | 1 | 1.0000 | nein |
| H1: Budget 256 vs. 1024 (D, generische Intervention) | 4 | 25.0% | 50.0% | 0 | 1 | 1.0000 | nein |
| H1: Budget 256 vs. 1024 (E, finanzspezifische Intervention) | 4 | 25.0% | 75.0% | 0 | 2 | 0.5000 | nein |
| H3: D vs. E bei Budget 256 | 4 | 25.0% | 25.0% | 0 | 0 | 1.0000 | nein |
| H3: D vs. E bei Budget 1024 | 4 | 50.0% | 75.0% | 0 | 1 | 1.0000 | nein |