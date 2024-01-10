# BwUniCluster2.0 Infos

## Nutzung via Terminal
### Login via SSH Client
Alle Anweisungen gibt es auch im [Wiki](https://wiki.bwhpc.de/e/BwUniCluster2.0/Login).
Für Windows wird als SSH Client MobaXterm empfohlen.

Bevor wir uns einloggen können muss jedoch [2FA](https://wiki.bwhpc.de/e/Registration/2FA) aktiviert sein und ein [Service Passwort](https://wiki.bwhpc.de/e/Registration/Password) gesetzt werden.
- Auf [diesem Link](https://login.bwidm.de/) kann man sich mit seinem Universitäts-Konto anmelden und seinen Username auslesen, sowie ein Service Passwort setzen.
- Auf [diesem Link](https://login.bwidm.de/user/twofa.xhtml) aktiviert man 2FA (hoffe ich zumindest, ist schon etwas her).

Nachdem wir das getan haben können wir in MobaXterm folgende Felder ausfüllen. Das sieht dann ca. so aus:
```
Remote name              : bwunicluster.scc.kit.edu    # or uc2.scc.kit.edu
Specify user name        : <username>
Port                     : 22
```

Man kann dort auch sein Service Passwort hinterlegen, wenn man es nicht jedes mal neu eingeben will.

Jetzt öffnet sich ein Terminal und man muss eigentlich nur noch ein OTP vom 2FA eingeben.

## Batch System
### Nützliche Links
Jobs kann man mit dem Batch System konfigurieren und in Auftrag geben. Sie werden dann ausgeführt sobald Resourcen frei sind.
Dazu gibt es viele Informationen. Hier einige verlinkt:
- [Slides](https://indico.scc.kit.edu/event/2667/attachments/4974/7529/05_2022-04-07_bwHPC_course_-_intro_batch_system.pdf)
- [Slurm](https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm)
- [Batch Queue](https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues)

### Batch Beispiel
Wir können ein .sh Skript schreiben, welches die gewünschte Konfiguration enthält:
```
#!/bin/bash
#SBATCH --partition=gpu_4_h100
#SBATCH --time=12:00:00
#SBATCH --mem=80000
#SBATCH --job-name=example
#SBATCH --gres=gpu:1

source ./venv/bin/activate
bash ./run_wos.sh
``` 
Dabei ist der partition Parameter eine Pflichtangabe.
Wie genau man am besten konfiguriert ist aber eine andere Geschichte.

Nach der Konfiguration können wir die gewünschten Befehle ausführen. In meinem Fall das Virtual Environment aktivieren und dann ein weiteres Skript ausführen.

Um den Job jetzt zu starten müssen wir 'sbatch script_name.sh' ausführen.
Mit 'squeue' können wir hierbei unsere Jobs in der Queue sehen.
