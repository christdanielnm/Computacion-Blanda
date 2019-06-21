# -*- coding: utf-8 -*-
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
lluvia = ctrl.Antecedent(np.arange(0, 101, 1), 'lluvia')
hora = ctrl.Antecedent(np.arange(0, 24, 1), 'hora')
estado_carretera = ctrl.Antecedent(np.arange(0, 101, 1), 'estado_carretera')
dia = ctrl.Antecedent(np.arange(0, 8, 1), 'dia')
mes = ctrl.Antecedent(np.arange(0, 13, 1), 'mes')
riesgo = ctrl.Consequent(np.arange(0, 101, 1), 'riesgo')

names=['bajo','medio','alto']
horas=['mañana','mediodia','tarde','noche']
carreteras=['buena','regular','mala']
dias=['semana','fin de semana']
meses=['cuatrimestre1','cuatrimestre2','cuatrimestre3']
lluvias=['poca','normal','intensa']

# Auto-membership function population is possible with .automf(3, 5, or 7)
hora.automf(names=horas)
estado_carretera.automf(names=carreteras)
dia.automf(names=dias)
mes.automf(names=meses)
riesgo.automf(names=names)
lluvia.automf(names=lluvias)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
lluvia['poca'] = fuzz.trapmf(lluvia.universe, [0, 0, 0, 30])
lluvia['normal'] = fuzz.trapmf(lluvia.universe, [20, 40, 50, 70])
lluvia['intensa'] = fuzz.trapmf(lluvia.universe, [50, 80, 100, 101])

hora['mañana'] = fuzz.trapmf(hora.universe, [0, 0, 1, 3]) # Bajo
hora['mediodia'] = fuzz.trapmf(hora.universe, [1, 7, 9, 13]) # Alto
hora['tarde'] = fuzz.trapmf(hora.universe, [12, 18, 20, 22]) # Normal
hora['noche'] = fuzz.trapmf(hora.universe, [21, 23, 23, 23]) # Muy bajo

estado_carretera['buena'] = fuzz.trapmf(estado_carretera.universe, [0, 0, 15, 30])
estado_carretera['regular'] = fuzz.trapmf(estado_carretera.universe, [25, 35, 60, 70])
estado_carretera['mala'] = fuzz.trapmf(estado_carretera.universe, [60, 75, 100, 100])

dia['semana'] = fuzz.trapmf(dia.universe, [0, 0, 0.5, 7]) 
dia['fin de semana'] = fuzz.trapmf(dia.universe, [4, 5.5, 7, 7])

mes['cuatrimestre1'] = fuzz.trapmf(mes.universe, [0, 0, 0.5, 4.5]) 
mes['cuatrimestre2'] = fuzz.trapmf(mes.universe, [3.5, 5.5, 7.5, 8.5])
mes['cuatrimestre3'] = fuzz.trapmf(mes.universe, [8.1, 11.5, 12, 12])

riesgo['bajo'] = fuzz.trapmf(riesgo.universe, [0, 0, 15, 35]) 
riesgo['medio'] = fuzz.trapmf(riesgo.universe, [25, 40, 60, 75])
riesgo['alto'] = fuzz.trapmf(riesgo.universe, [70, 85, 100, 100])

rule1 = ctrl.Rule(lluvia['poca'] & estado_carretera['buena'],  riesgo['bajo'])
rule2 = ctrl.Rule(lluvia['poca'] & estado_carretera['regular'],  riesgo['bajo'])
rule3 = ctrl.Rule(lluvia['poca'] & estado_carretera['mala'],  riesgo['medio'])
rule4 = ctrl.Rule(lluvia['normal'] & estado_carretera['buena'], riesgo['medio'])
rule5 = ctrl.Rule(lluvia['normal'] & estado_carretera['regular'], riesgo['medio'])
rule6 = ctrl.Rule(lluvia['normal'] & estado_carretera['mala'], riesgo['medio'])
rule7 = ctrl.Rule(lluvia['intensa'] & estado_carretera['buena'], riesgo['medio'])
rule8 = ctrl.Rule(lluvia['intensa'] & estado_carretera['regular'], riesgo['alto'])
rule9 = ctrl.Rule(lluvia['intensa'] & estado_carretera['mala'], riesgo['alto'])
rule10= ctrl.Rule(hora['mañana'] & estado_carretera['buena'], riesgo['bajo'])
rule11= ctrl.Rule(hora['mañana'] & estado_carretera['regular'], riesgo['medio'])
rule12= ctrl.Rule(hora['mañana'] & estado_carretera['mala'], riesgo['medio'])
rule13= ctrl.Rule(hora['mediodia'] & estado_carretera['buena'], riesgo['bajo'])
rule14= ctrl.Rule(hora['mediodia'] & estado_carretera['regular'], riesgo['medio'])
rule15= ctrl.Rule(hora['mediodia'] & estado_carretera['mala'], riesgo['medio'])
rule16= ctrl.Rule(hora['tarde'] & estado_carretera['buena'], riesgo['bajo'])
rule17= ctrl.Rule(hora['tarde'] & estado_carretera['regular'], riesgo['bajo'])
rule18= ctrl.Rule(hora['tarde'] & estado_carretera['mala'], riesgo['medio'])
rule19= ctrl.Rule(hora['noche'] & estado_carretera['buena'], riesgo['medio'])
rule20= ctrl.Rule(hora['noche'] & estado_carretera['regular'], riesgo['medio'])
rule21= ctrl.Rule(hora['noche'] & estado_carretera['mala'], riesgo['alto'])

#rule3.view()

#nivel_riesgo.compute()
#print("Riesgo, " + str(nivel_riesgo.output['riesgo']))
#acc= nivel_riesgo.output['riesgo']

def calcular_Riesgo(l,c,h):
    riesgo_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, 
                                  rule8, rule9, rule10, rule11, rule12, rule13,rule14,
                                  rule15,rule16,rule17,rule18,rule19,rule20,rule21])
    nivel_riesgo = ctrl.ControlSystemSimulation(riesgo_ctrl)
    nivel_riesgo.input['lluvia'] = l
    nivel_riesgo.input['estado_carretera'] = c
    nivel_riesgo.input['hora'] = h
    nivel_riesgo.compute()
    return nivel_riesgo.output['riesgo']
    #print("Riesgo, " + str(nivel_riesgo.output['riesgo']))
    #riesgo.view(sim=nivel_riesgo)

#print(calcular_Riesgo(55,36))
    
    