# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:10:59 2016

@author: lsantella
"""

import pandas as pd
import numpy as np
import matplotlib as mp
from sklearn import preprocessing

names = ['No-Name', 'Periodo', 'Cedula', 'F_Nacimiento', 'Edad', 'E_Civil',
         'Sexo', 'Escuela', 'A_Ingreso', 'Mod_Ingreso', 'S_Cursa', 'C_Direccion',
         'Motivo_C', 'M_Inscritas', 'M_Aprobadas', 'M_Retiradas', 'M_Reprobadas',
         'Promedio_Pond', 'Eficiencia', 'Reprobacion_Motiv', 'Materias_Actuales', 
         'Tesis', 'Veces_Tesis', 'Procedencia', 'Residencia', 'Acompanantes', 
         'Tipo_Vivienda', 'Monto_Alquiler', 'Direccion_Res_Alq', 'Matrimonio', 
         'Otro_Beneficio', 'F_Motiv_Sol_Benef', 'Actividad_Ingresos', 'Tipo_Frec_Actividad',
         'Monto_Mensual_Beca', 'Mensual_Aporte_Resp', 'Mesual_Aporte_Amigos', 'Aporte_Destajo',
         'Mensual_Ingreso_Total', 'Alimentacion', 'T_Publico', 'Gastos_Medicos',
         'Gastos_Odonto', 'Gastos_Personales', 'Gastos_Alquiler','Gastos_Materiales',
         'Gastos_Recreacion', 'Gastos_Otros', 'Total_Egresos', 'Responsable_Economico', 
         'Carga_Familiar', 'Ingreso_Responsable', 'Ingresp_Resp_Otros','Total_Ingreso_Resp', 
         'Vivienda', 'Gastos_ALimentacion_Resp', 'Gastos_Transporte_Resp','Gastos_Medicos_Resp', 
         'Gastos_Odonto_Resp', 'Gastos_Educativos_Resp', 'Gastos_SP_Gas', 'Condominio_Resp', 
         'Otros_Gastos_Resp', 'Total_Egresos_Resp', 'Opinion_Usuarios', 'Sugerencia_Recomendaciones']

'''
Cargamos los datos proporcionados
'''         
data = pd.read_csv('./data.csv', 
                   sep=',',
                   skiprows = 1,
                   names = names)
                   
'''
    Sustitucion de los valores vacios por NaN ya que es
    consistente con el paquete pandas y tambien por el
    paquete sklearn
'''
data = data.replace('', np.nan)
data = data.replace(r'NA', np.nan)

'''
    Transformaremos todos los datos a un solo tipo de dato.
    En este caso, trabajaremos con datos numericos, debido
    a las ventajas en cuanto a eficiencia y portabilidad
    a la hora de aplicar algoritmos de analisis de datos
'''
unique_values = pd.unique(data['Periodo'])

data['Periodo'] = data['Periodo'].replace(

data['Periodo']
data.to_csv('./minable.csv',
            sep=',')

