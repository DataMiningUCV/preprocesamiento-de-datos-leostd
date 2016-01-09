# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:10:59 2016

@author: lsantella
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random as rn
import re
from sklearn import preprocessing

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
matplotlib.style.use('ggplot')
rn.seed(2016)


        
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

'''
    ********** PERIODO ***********
'''

'''
    Los valores de la columna Periodo, fueron separados en 2 columnas
    A_Periodo, es el anho (2015, 2014) y N_Periodo, es el numero del 
    periodo(I, II)
'''
data['Periodo'] = data['Periodo'].str.lower()
data['Periodo']
data['A_Periodo'] = 'miss'
data['A_Periodo'] = np.where(data['Periodo'].str.contains(r'14'), '2014', data['A_Periodo'])
data['A_Periodo'] = np.where(data['Periodo'].str.contains(r'15'), '2015', data['A_Periodo'])
data['A_Periodo'] = np.where(data['Periodo'].str.contains(r'2015-2016'), '2015', data['A_Periodo'])
data['A_Periodo'] = np.where(data['Periodo'].str.contains(r'2014-2015'), '2014', data['A_Periodo'])
data['A_Periodo'].iloc[63] = '2015'
data[['Periodo','A_Periodo']]
data['N_Periodo'] = 'miss'
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'i'), 'I', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'ii'), 'II', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'-1'), 'I', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'1s'), 'I', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r's1'), 'I', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'2s'), 'II', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'-01'), 'I', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'2014-2015'), 'II', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'-02'), 'II', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'pri'), 'I', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'seg'), 'II', data['N_Periodo'])
data['N_Periodo'] = np.where(data['Periodo'].str.contains(r'sec'), 'II', data['N_Periodo'])
data['N_Periodo'].loc[[86,131,62,50,4]] = 'II'
data['A_Periodo'] = np.where(data['A_Periodo'].str.contains('nan'), np.nan, data['A_Periodo'])
data['N_Periodo'] = np.where(data['N_Periodo'].str.contains('nan'), np.nan, data['N_Periodo'])
#data['N_Periodo'].loc[[2,3,32,48,73,118,124,132,148,160,180]] = np.nan




'''
    Imputacion de los datos faltantes.
    En la columna A_Periodo se imputo la moda (2014)
    En la columna N_Periodo, se imputo el valor que mas repite segun el valor 
    de A_Periodo en esa fila.
'''
v1 = len(data[data['A_Periodo'] == '2015'])
v2 = len(data[data['A_Periodo'] == '2014'])

for index, row in data.iterrows():
    if row['A_Periodo'] == 'miss':
        if rn.random() <= (v1/(v1+v2)):
            data.loc[index, 'A_Periodo'] = '2015'
        else:
            data.loc[index, 'A_Periodo'] = '2014'

#group_1 = data[['A_Periodo', 'N_Periodo']].groupby(['N_Periodo', 'A_Periodo'])
#group_1.count()

data['N_Periodo'] = np.where((data['A_Periodo'] == '2014') & (data['N_Periodo'] == 'miss'), 'II', data['N_Periodo'])
data['N_Periodo'] = np.where((data['A_Periodo'] == '2015') & (data['N_Periodo'] == 'miss'), 'I', data['N_Periodo'])

'''
    ********** FECHA DE NACIMIENTO ***********
'''
data['F_Nacimiento'] = data['F_Nacimiento'].str.replace('-', '/')
data['F_Nacimiento'] = data['F_Nacimiento'].str.replace(' ', '/')
data['F_Nacimiento'].loc[42] = '18/05/1992'
data['F_Nacimiento'].loc[151] = '21/04/1994'
data['F_Nacimiento'].loc[19] = '22/04/1985'

'''
    Este snippet convierte a las fechas que tienen anhos de 2 cifras
    por ejemplo, 10/04/95, las convierte en anhos de 4 cifras, de 
    esta manera tenemos un formato uniforme en toda la columna
'''    
for index, row in data['F_Nacimiento'].iteritems():
    date = re.search('(?P<day>\d\d)/(?P<month>\d\d)/(?P<year>\d\d{1,3})$',row)
    if date and len(date.group(3)) < 4:
        data.loc[index, 'F_Nacimiento']= date.group(1) +    \
                                        '/'+ date.group(2) + \
                                        '19' + date.group(3)
'''
    ********** EDAD ***********
'''

'''
    Esta columna solo presenta algunas filas con la palabra 'anhos'. 
    Al eliminar las instancias de dicha palabra, quedara una columna
    en un formato uniforme
'''

data['Edad'] = data['Edad'].str.lower()
data['Edad'] = data['Edad'].str.replace('[añÑos\s]+', '')

'''
    ********** ESTADO CIVIL ***********
'''

data['E_Civil'] = data['E_Civil'].str.replace('S\w+', '1')
data['E_Civil'] = data['E_Civil'].str.replace('U\w+', '2')
data['E_Civil'] = data['E_Civil'].str.replace('C\w+', '3')
data['E_Civil'] = data['E_Civil'].str.replace('V\w+', '4')

'''
    ********** SEXO ***********
'''

'''
    Masculino: 1     Femenino: 2
'''
data['Sexo'] = data['Sexo'].str.replace('M\w+', '1')
data['Sexo'] = data['Sexo'].str.replace('F\w+', '2')

'''
    ********** ESCUELA ***********
'''

'''
    Enfermeria: 1     Bioanalisis: 2
'''
data['Sexo'] = data['Sexo'].str.replace('B\w+', '1')
data['Sexo'] = data['Sexo'].str.replace('E\w+', '2')

'''
    ********** MODO DE INGRESO ***********
'''

'''
    Convenios Interinstitucionales (nacionales e internacionales): 1
    Prueba Interna y/o propedéutico: 2
    Convenios Internos (Deportistas, artistas, hijos empleados docente y obreros, Samuel Robinson): 3
    Asignado OPSU: 4
'''


data.to_csv('./minable.csv',
            sep=',')