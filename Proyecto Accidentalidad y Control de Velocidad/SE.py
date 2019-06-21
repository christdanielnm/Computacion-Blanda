from pyDatalog import pyDatalog
from Fuzzy2 import *
pyDatalog.create_terms('Velocidad, V, Accidentalidad, Velocidad_Sugerida')

#Base del conocimiento
+Velocidad(0,"80-90")
+Velocidad(1,"60-70")
+Velocidad(2,"40-50")
+Velocidad(3,"20-30")

#Reglas
V(Accidentalidad, Velocidad_Sugerida) <= Velocidad(Accidentalidad, Velocidad_Sugerida)

#print(V(Accidentalidad, Velocidad_Sugerida))
#Falta retornar
