import streamlit as st
import pandas as pd
import numpy as np
from os import path

st.set_page_config(
    page_title="Machine learning",
    page_icon="🚲",
)

st.title("Prädiktive Analyse von Fahrradunfällen basierend auf Beinahe-Unfällen und realen Unfällen aus dem Jahr 2022")

st.markdown("### 1. Überblick über die Projektaufgabe")
st.markdown("""
    Im Rahmen unseres Projekts zur prädiktiven Analyse von Fahrradunfällen untersuchen wir Beinahe-Unfälle und reale Unfälle, 
    die im Jahr 2022 erfasst wurden. Ziel ist es, Risikofaktoren für Fahrradunfälle zu identifizieren und darauf aufbauend Maßnahmen 
    zur Verbesserung der Verkehrssicherheit für Radfahrer zu entwickeln. Durch die Kombination der verschiedenen Datenquellen und die Nutzung maschineller
     Lernmethoden sollen Gefahrenstellen vorhergesagt und präventive Strategien entwickelt werden."""
)


st.markdown("### 2. Motivation und Relevanz")
st.markdown(""" Unsere Motivation ist es, die Sicherheit von Radfahrern zu erhöhen und gleichzeitig das Vertrauen in das Fahrrad als umweltfreundliches und gesundes Fortbewegungsmittel zu stärken.
             Wir möchten eine App oder eine Funktion in Google Maps entwickeln, die sowohl für Radfahrer als auch für Autofahrer auf besonders gefährliche Straßenabschnitte aufmerksam macht, um Unfälle zu verhindern. 
""")

st.markdown("### 3. Beschreibung der verwendeten Daten und Methodik ")
st.markdown("### 3. 1 Datensätze")

st.markdown(""" Für unsere Analyse haben wir zwei Haupt-Datensätze verwendet. Der eine ist ein  Datensatz zu Beinahe-Unfällen und 
            der andere zu realen Unfällen. Bei Beinahe-Unfällen handelt es sich um Situationen, in denen es fast zu Unfällen gekommen ist. 
            Man kann aus dem Datensatz  Beinahe-Unfälle wertvolle Hinweise auf risikobehaftete Straßenabschnitte und Verhaltensweisen ermitteln. 
            Da beinahe Unfälle häufiger passieren als reale Unfälle bieten die eine breitere Datenbasis für statistische Analysen.
            
            
    Die Beinahe-Unfälle haben wir von der Simra(Sicherheit im Radverkehr) Projekt. 
            Im Rahmen des SimRa-Projekts wurden Daten darüber gesammelt, wo sich in der Stadt Gefahren für Radfahrer häufen, 
            welcher Art diese sind, ob sie zeitlich oder örtlich gehäuft auftreten und wo die Haupt Verkehrsströme auf dem Rad liegen. 
            Im Rahmen des Projekts wurde eine Smartphone-App entwickelt, die Routen mit Hilfe von GPS-Daten aufzeichnet und mit Hilfe von
            Beschleunigungssensoren gefährliche Situationen erkennt - wie plötzliches Bremsen, Ausweichen oder sogar einen Sturz.
            Nach der Fahrt werden die Radfahrer gebeten, diese erkannten Gefahrensituationen zu kategorisieren und zu kommentieren,
            nicht erkannte Gefahrensituationen hinzuzufügen und einen Upload auf die Projektserver zu genehmigen.
            
""")

data = pd.read_csv('../Berlin-incidents.csv', on_bad_lines='skip') #path folder of the data file

st.write(data) 
st.write("Beinahunfall tabelle")    

st.map(data)

st.markdown (""" Als zweiten Datensatz haben wir reale Unfälle genommen. Der Datensatz stammt aus den offiziellen Aufzeichnungen der Verkehrsbehörden (Unfallatlas Deutschland). 
            Die realen Unfalldatensatz hingegen liefern uns konkrete Informationen über tatsächlich geschehene Unfälle, einschließlich die Größe des Unfalls, 
            der beteiligten Verkehrsteilnehmer und die Umständen davon. Da uns hier konkrete Ursachen zum Beispiel wie Geschwindigkeitsüberschreitung, 
            Ablenkung und Mangel am Fahrzeug ist, ist es möglich, eine genaue Ursachenanalyse zu unternehmen.
            
            """)


data = pd.read_csv('../Unfallorte2022_LinRef.csv', sep=",", on_bad_lines='skip') #path folder of the data file

st.write(data) 
st.write("Unfallorte tabelle 2022")    

#st.map(data)

st.markdown("### 3. 2 Methodik")

st.markdown("""
            Der Prozess der Vorhersage von Fahrradunfällen war keine einfache Aufgabe. Wir begannen damit, tief in den Datensatz einzutauchen und zu versuchen,
             ihn so gut wie möglich zu verstehen, indem wir mehrere Fragen, wie die Bedeutung jeder Spaltenüberschrift, welche Art von Werten in der Spalte stehen und wie oft sie vorkommen.
Während des Prozesses wurden viele Spalten umbenannt, damit sie für unsere Leser leicht verständlich sind, und wir haben auch einige Funktionen entwickelt, z. B. haben wir die Uhrzeit, das Datum usw. aus der Spalte Zeitstempel (ts) im Datensatz für Beinahe-Unfälle erzeugt. Danach haben wir einige Spalten entfernt, die keinen großen Einfluss auf die Unfälle haben, und somit auch nicht auf die Vorhersagen haben.

""")






st.markdown(""" Um die Unfallanalyse gezielt auf Berlin auszurichten, wurden zunächst die entsprechenden Daten aus einer CSV-Datei geladen. Dies geschah mit Hilfe der Bibliothek pandas. 
            Zuerst wurde der Pfad zur Datei Unfallorte2022_LinRef.csv definiert und anschließend die Datei in ein DataFrame eingelesen. 
            Der Datensatz enthält 256.492 Einträge, die verschiedene Merkmale wie Ortsangaben, Zeitpunkte, Lichtverhältnisse und Verkehrsteilnehmer
              (z.B. Radfahrer, PKW-Fahrer) der jeweiligen Unfälle umfassen. """)



st.markdown("### 4. Datenanalyse ")
#st.markdown("### 4.1 Datenaufbereitung und Visualisierung der Unfallorte in Berlin")
st.markdown("### 4.1 Datenaufbereitung Unfall-Datensatz")

st.markdown("""
    
Verschiedene Merkmale der Unfälle hinsichtlich ihrer Häufigkeit wurde analysiert und in Form von Histogrammen dargestellt.
Die ausgewählten Spalten umfassten unter anderem den Monat des Unfalls (UMONAT), die Uhrzeit (USTUNDE), den Wochentag (UWOCHENTAG) sowie die Lichtverhältnisse
(ULICHTVERH). Ziel dieser Analyse war es, herauszufinden, ob bestimmte Zeiträume oder Bedingungen mit einer erhöhten Unfallhäufigkeit für Radfahrer korrelieren.

""")

st.image("1.jpeg")
st.image("2.jpeg")
st.image("3.jpeg")
st.image("4.jpeg")
st.image("5.jpeg")
st.image("11.jpeg")
st.image("12.jpeg")

st.markdown("""
Darüber hinaus wurde eine Einteilung in Gefahrenklassen vorgenommen, basierend auf den oben genannten Merkmalen. Jede dieser Kategorien wurde nach Risiko in "leicht", "mittel" und "schwer" klassifiziert. Zum Beispiel gelten die Monate Januar und Februar als "leicht", während die Sommermonate aufgrund des höheren Verkehrsaufkommens als "schwer" eingestuft wurden. Ebenso wurden Uhrzeiten, Wochentage, Lichtverhältnisse und Straßenbedingungen entsprechend kategorisiert. Hier ist die Tabelle dazu:
            """)
st.image("tabelle.png")

st.markdown("### 4.2 Visualisierung ")
st.markdown("""
Zur Visualisierung der Unfalldaten wurde die Bibliothek Geopandas benutzt, um die geografischen Unfallorte auf Karten darzustellen. Die bereits in den vorherigen Schritten gefilterten und bereinigten Daten wurden in ein GeoDataFrame konvertiert, um die räumliche Komponente der Unfalldaten zu berücksichtigen.
Zunächst wurden die Koordinaten, die in den Spalten LINREFX und LINREFY gespeichert sind, überprüft und, falls nötig, von String- in das Format float64 umgewandelt. Diese Umwandlung war notwendig, um die Koordinaten in geometrische Punkte zu konvertieren, die für die Visualisierung auf Karten erforderlich sind. Nach der Konvertierung wurde mithilfe der Shapely-Bibliothek eine geometrische Spalte aus den Koordinatenpunkten erstellt, die als Basis für die geodätische Analyse diente.
Außerdem wurde eine einfache Visualisierung mit Matplotlib erstellt, um die Verteilung der Unfallorte basierend auf der Regionsbezeichnung darzustellen. Diese Visualisierungen, sowohl in Folium als auch in Matplotlib, bieten eine umfassende und benutzerfreundliche Möglichkeit an, die Unfalldaten in Berlin zu analysieren bzw. zu kennzeichnen.

            """)

st.image("6.jpeg")
st.image("7.jpeg")
#st.image("8.jpeg")
st.image("9.jpeg")


st.markdown("### 5. Beschreibung der Ergebnissen")
st.markdown("""
            
Das Modell des maschinellen Lernens, für das wir uns entschieden haben, ist die Klassifizierung. Es wurde dem Beinahunfall-Datensatz die Spalte “Scary” als Zielvariable ausgewählt. Bei dem Unfall-Datensatz wurde eine Spalte “risk_level” erstellt und als Zielvariable ausgewählt, welche aus den Ergebnissen von 4.1 Datenaufbereitung Unfall-Datensatz ausgewertet wurde.
Die Ergebnisse des Modells für den Unfalldatensatz zeigen eine Genauigkeit von 1.0 (100 %), was darauf hindeutet, dass das Modell bei allen Datenpunkten erfolgreich funktioniert hat. Es gibt jedoch Zweifel an dieser hohen Genauigkeit, da das Analyse-Diagramm nur einen Balken für „Kategorie“ zeigt, während andere Merkmale nicht sichtbar sind. Dies könnte darauf hindeuten, dass das Modell möglicherweise nicht alle relevanten Einflussfaktoren angemessen berücksichtigt.
Ebenso könnte dies ein Hinweis auf ein sogenanntes “Overfitting” sein, da eine Genauigkeit von 1 sehr unwahrscheinlich für ein Modell ist. Das Ergebnis aus dem Datensatz für Beinahe-Unfälle lag dagegen bei 0.74, welches realistisch von Zahl her ist. 

            """)
st.image("unfall_perf.png", caption="Unfalldatensatz performance")
st.image("performance.png", caption="Beinahunfall datensatz performance")
st.image("feature importance.png")


st.markdown("### 6. Reflexion")
st.markdown("""
Zu Beginn waren wir begeistert von der Aussicht, Fahrradunfälle anhand der verfügbaren Daten vorhersagen zu können. Während der Datenexploration begannen wir jedoch, uns zu fragen, was der eigentliche Machine-Learning-Aspekt unseres Projekts ist und wie wir ihn effektiv zur Lösung unserer Problemstellung einsetzen können. Dies lag vor allem daran, dass unsere Ergebnisse zu diesem Zeitpunkt wenig oder gar nichts mit Machine Learning zu tun hatten. In Anbetracht dessen kam der Gedanke auf, dass möglicherweise kein maschinelles Lernen erforderlich sei und dass die Fragestellung durchaus mit statistischen Methoden gelöst werden könnte.

Trotz dieser Bedenken entschieden wir uns, verschiedene Modelle und Ideen zu testen, um das Potenzial von Machine Learning zu erkunden. Der Einfachheit halber haben wir beschlossen, jeden Datensatz einzeln zu analysieren, um ein besseres Verständnis für die spezifischen Merkmale und Muster in den Daten zu erlangen.

Für die Zukunft planen wir, die Datensätze zusammenzuführen und eine umfassendere Analyse durchzuführen. Durch die Kombination der Daten können wir wertvolle Erkenntnisse gewinnen und die Ergebnisse vergleichen, um Ähnlichkeiten und Unterschiede zu identifizieren. Dies könnte nicht nur unsere Vorhersagemodelle verbessern, sondern auch neue Perspektiven auf die Risikofaktoren von Fahrradunfällen eröffnen. Darüber hinaus wäre es sinnvoll, verschiedene Machine-Learning-Algorithmen zu evaluieren, um die beste Methode zur Vorhersage von Fahrradunfällen zu ermitteln. Wir selber hatten die Idee, eine Art Feature in Navigationssysteme zu integrieren. Wenn es auf Fahrradfahrer steht, dass es dann wie bei dem Auto, beim Stau, bei dem Fahrradfahrer, dann eine Art Farbkennung oder Beep Geräusche tauchen und Sie vor Gefahren bewahren. 
""")
