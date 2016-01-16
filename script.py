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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
matplotlib.style.use('ggplot')
rn.seed(2016)
	
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
    
def is_not_number(x):
    try:
        float(x)
        return False
    except ValueError:
        pass
    
    try:
        import unicodedata as uc
        uc.numeric(x)
        return False
    except (TypeError, ValueError):
        pass
    return True
        
names = ['No-Name', 'Periodo', 'Cedula', 'F_Nacimiento', 'Edad', 'E_Civil',
         'Sexo', 'Escuela', 'A_Ingreso', 'Mod_Ingreso', 'S_Cursa', 'C_Direccion',
         'Motivo_C', 'M_Inscritas', 'M_Aprobadas', 'M_Retiradas', 'M_Reprobadas',
         'Promedio_Pond', 'Eficiencia', 'Reprobacion_Motiv', 'Materias_Actuales', 
         'Tesis', 'Veces_Tesis', 'Procedencia', 'Residencia', 'Acompanantes', 
         'Tipo_Vivienda', 'Monto_Alquiler', 'Direccion_Res_Alq', 'Matrimonio', 
         'Otro_Beneficio', 'F_Motiv_Sol_Benef', 'Actividad_Ingresos', 'Tipo_Frec_Actividad',
         'Monto_Mensual_Beca', 'Mensual_Aporte_Resp', 'Mensual_Aporte_Amigos', 'Aporte_Destajo',
         'Mensual_Ingreso_Total', 'Alimentacion', 'T_Publico', 'Gastos_Medicos',
         'Gastos_Odonto', 'Gastos_Personales', 'Gastos_Alquiler','Gastos_Materiales',
         'Gastos_Recreacion', 'Gastos_Otros', 'Total_Egresos', 'Responsable_Economico', 
         'Carga_Familiar', 'Ingreso_Responsable', 'Ingreso_Resp_Otros','Total_Ingreso_Resp', 
         'Vivienda', 'Gastos_Alimentacion_Resp', 'Gastos_Transporte_Resp','Gastos_Medicos_Resp', 
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
data['F_Nacimiento'].loc[151] = '21/03/1993'
data['F_Nacimiento'].loc[19] = '22/04/1985'
data['F_Nacimiento'].loc[12] = '10/05/1989'

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
                                        '/19' + date.group(3)
data.F_Nacimiento                                        
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
data['E_Civil'] = data['E_Civil'].str.replace('[(a)]+', '')
data['E_Civil']
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
data['Escuela'] = data['Escuela'].str.replace('B\w+á\w+', '1')
data['Escuela'] = data['Escuela'].str.replace('E\w+í\w+', '2')
#data['Escuela'] = data['Escuela'].str.replace('í|á\w+', '')
data.Escuela
'''
    ********** MODO DE INGRESO ***********
'''

'''
    Convenios Interinstitucionales (nacionales e internacionales): 1
    Prueba Interna y/o propedéutico: 2
    Convenios Internos (Deportistas, artistas, hijos empleados docente y obreros, Samuel Robinson): 3
    Asignado OPSU: 4
'''
data.loc[0, 'Mod_Ingreso'] = '1'
ind = data.Mod_Ingreso.str.contains('Prueba')
data.loc[ind, 'Mod_Ingreso'] = '2'
ind = data.Mod_Ingreso.str.contains('Convenios')
data.loc[ind, 'Mod_Ingreso'] = '3'
data['Sexo'] = data['Sexo'].str.replace('E\w+', '2')
ind = data.Mod_Ingreso.str.contains('Asignado')
data.loc[ind, 'Mod_Ingreso'] = '4'
data['Mod_Ingreso']

'''
    ********** SEMESTRE QUE CURSA ***********
'''

col = 'S_Cursa'
for index, row in data[col].iteritems():
    x = re.search('(?P<number>\d+)', row)
    if (x):
        data.loc[index, col] = x.group(1)
data[col] = data[col].astype(int)
'''
    ********** CAMBIO DE DIRECCION ***********
'''
col = 'C_Direccion'
data[col] = np.where(data[col] == 'Si', '1', '0')
data[col]

'''
    ********** MATERIAS_APROBADAS EL SEMESTRE ANTERIOR ***********
'''
col = 'M_Aprobadas'
data[col].loc[1] = data[col].mode().loc[0]

'''
    ********** PROMEDIO PONDERADO ***********
'''
col = 'Promedio_Pond'
for index, row in data[col].iteritems():
    if row > 20:
        x = row
        while x > 20:
            x /= 10;
        data.loc[index, col] = x

'''
    ********** EFICIENCIA ***********
'''
col = 'Eficiencia'
for index, row in data[col].iteritems():
    if row > 1:
        x = row
        while x > 1:
            x /= 10;
        data.loc[index, col] = x 
'''
    ********** Tesis ***********
'''
col = 'Tesis'
data[col] = np.where(data[col] == 'Si', '1', '0')

'''
    ********** VECES_TESIS (PASANTIAS, ETC) ***********
'''

'''
    Primera vez:  1     Segunda vez:  2      Mas de dos: 3
'''
col = 'Veces_Tesis'
data[col] = data[col].fillna('0')
data[col] = data[col].str.replace('Primera vez', '1')
data[col] = data[col].str.replace('Segunda vez', '2')
data[col] = data[col].str.replace('Más de dos', '3')

'''
    ********** PROCEDENCIA ***********
'''

col = 'Procedencia'
unique_values = data[col].unique()
unique_values.T
cant = np.array(range(len(unique_values)))
change = dict(zip(unique_values.T, cant))
data.replace({col:change}, inplace = True)
data[col]

'''
    ********** RESIDENCIA ***********
'''

col = 'Residencia'
moda = data[col].mode().loc[0]
data[col] = data[col].fillna(moda)
unique_values = data[col].unique()
unique_values.T
cant = np.array(range(len(unique_values)))
change = dict(zip(unique_values.T, cant))
data.replace({col:change}, inplace = True)
data[col]

'''
    ********** ACOMPANANTES ***********
'''

col = 'Acompanantes'
data[col].unique()
#data[col].str.capitalize()
for index, row in data[col].iteritems():
    data.loc[index, col] = row.capitalize()
data[col] = data[col].str.replace('\(a\)|\(as\)', '')
unique_values = data[col].unique()
unique_values.T 
cant = np.array(range(len(unique_values)))
change = dict(zip(unique_values.T, cant))
data.replace({col:change}, inplace = True)
data[col]

'''
    ********** TIPO DE VIVIENDA ***********
'''

col = 'Tipo_Vivienda'
for index, row in data[col].iteritems():
    data.loc[index, col] = row.capitalize()
unique_values = data[col].unique()
cant = np.array(range(len(unique_values)))
change = dict(zip(unique_values.T, cant))
data.replace({col:change}, inplace = True)
data[col]

'''
    ********** MONTO ALQUILER ***********
'''
col = 'Monto_Alquiler'
data.loc[115, col] = 150
data.loc[64, col] = 2700
data[col] = data[col].fillna(0)
data[col] = data[col].astype(float)
'''
    ********** MATRIMONIO ***********
'''
col = 'Matrimonio'
data[col] = np.where(data[col] == 'Si', '1', '0')

'''
    ********** OTRO BENEFICIO ***********
'''
col = 'Otro_Beneficio'
data[col] = np.where(data[col] == 'Si', '1', '0')

'''
    ********** ACTIVIDAD INGRESOS ***********
'''
col = 'Actividad_Ingresos'
data[col] = np.where( data[col] == 'Si', 1, 0)

'''
    ********** MONTO DE LA BECA MENSUAL ***********
'''
col = 'Monto_Mensual_Beca'
data[col].loc[142] = 1500

'''
    ********** INGRESOS ***********
'''
#data.rename(columns={'Mesual_Aporte_Amigos' : 'Mensual_Aporte_Amigos'}, inplace=True)
col = ['Mensual_Aporte_Resp', 'Mensual_Aporte_Amigos', 'Aporte_Destajo', \
       'Mensual_Ingreso_Total']   
data[col] = data[col].fillna(0)

'''
     ********** GASTOS ***********
'''
col = ['Alimentacion', 'T_Publico', 'Gastos_Medicos', \
       'Gastos_Odonto', 'Gastos_Personales', \
       'Gastos_Alquiler','Gastos_Materiales', \
       'Gastos_Recreacion', 'Gastos_Otros', 'Total_Egresos']
data[col] = data[col].fillna(0)
data[col] = data[col].astype(float)
'''
    ********** RESPONSABLE ECONOMICO ***********
'''

col = 'Responsable_Economico'
data[col].unique()
data[col] = data[col].str.replace('esposo', 'Esposo')
data[col] = data[col].str.replace('ninguno', 'Ninguno')
data[col] = data[col].str.replace('tia', 'Tia')
data[col] = data[col].str.replace('MI HERMANA|hermana', 'Hermana')
data[col] = data[col].str.replace('abuela', 'Abuela')
unique_values = data[col].unique()
cant = np.array(range(len(unique_values)))
change = dict(zip(unique_values.T, cant))
data.replace({col:change}, inplace = True)

'''
     ********** INGRESOS DEL RESPONSABLE ECONOMICO ***********
'''     
col = 'Ingreso_Responsable'
data[col] = data[col].str.replace('bs', '')
data.loc[37, col] = 9066.84
data.loc[37, col] = 9066.84
data.loc[105, col] = 16000
data.loc[142, col] = 55635.46
data.loc[187, col] = 7873.44
data.loc[37, col] = 9066.84
for index, row in data.loc[data[col].apply(is_not_number), col].iteritems():
    x = row
    print x
    x = re.search(' ',x)
    if( x ):
        x = row.replace(' ','')
        data.loc[index, col] = x
        row = x
    else:
        x = row
    x = re.search(',(?P<point>\d+$)', x)
    if ( x ):
       x = re.sub(',\d+$','.' + x.group(1), row)
       x = x.replace(',','')
       data.loc[index, col] = x
    else:
        x = row.replace(',','')
    if ( is_not_number(x) ):
        print x, 'lol'
    else:
        data.loc[index, col] = x
data[col]
'''
     ********** OTROS INGRESOS DEL RESPONSABLE ECONOMICO ***********
'''  

col = 'Ingreso_Resp_Otros'
data[col] = data[col].fillna(0)
data[col] = data[col].str.replace('bs','')
data.loc[180, col] = 0
data[col]

'''
     ********** TOTAL INGRESOS DEL RESPONSABLE ECONOMICO ***********
'''  

col = 'Total_Ingreso_Resp'
data[col] = data[col].str.replace('bs','')
for index, row in data.loc[data[col].apply(is_not_number), col].iteritems():
    x = row
    print x
    x = re.search(' ',x)
    if( x ):
        x = row.replace(' ','')
        data.loc[index, col] = x
        row = x
    else:
        x = row
    x = re.search(',(?P<point>\d+$)', x)
    if ( x ):
       x = re.sub(',\d+$','.' + x.group(1), row)
       x = x.replace(',','')
       data.loc[index, col] = x
    else:
        x = row.replace(',','')
    if ( is_not_number(x) ):
        print x, 'lol'
    else:
        data.loc[index, col] = x
'''
     ********** GASTOS DEL RESPONSABLE ECONOMICO ***********
'''
#data.rename(columns={'Gastos_ALimentacion_Resp':'Gastos_Alimentacion_Resp'}, inplace=True)
col = ['Vivienda', 'Gastos_Alimentacion_Resp', 'Gastos_Transporte_Resp',
       'Gastos_Medicos_Resp','Gastos_Odonto_Resp', 'Gastos_Educativos_Resp', 
       'Gastos_SP_Gas', 'Condominio_Resp', 'Otros_Gastos_Resp', 'Total_Egresos_Resp']
data[col] = data[col].fillna(0)
data[col] = data[col].replace('bs','')


'''
      ********** VIVIENDA ***********
'''
col = 'Vivienda'
not_numbers = data[col].apply(is_not_number)
for index, row in data.loc[not_numbers, col].iteritems():
    x = row.replace(' ','').replace(',','.')
    if( is_number(x) ):
        data.loc[index,col] = x
    else:
        data.loc[index,col] = 0
data[col] = data[col].astype(float)
data.loc[[142,5], col]

'''
      ********** GASTOS MEDICOS DEL RESPONSABLE ***********
'''
col = 'Gastos_Medicos_Resp'
data[col] = data[col].replace('bs','')
not_numbers = data[col].apply(is_not_number)
data.loc[not_numbers, col]
data.loc[105, col] = 3000
data[col] = data[col].astype(float)
data.info()

'''
      ********** GASTOS MEDICOS DEL RESPONSABLE ***********
'''
col = 'Gastos_Odonto_Resp'
data[col] = data[col].replace('bs','')
not_numbers = data[col].apply(is_not_number)
data.loc[not_numbers, col]
data.loc[96, col] = 0
data[col] = data[col].astype(float)
data.info()

'''
      ********** GASTOS DE CONDOMINIO DEL RESPONSABLE ***********
'''
col = 'Condominio_Resp'
data[col] = data[col].replace('bs','')
not_numbers = data[col].apply(is_not_number)
data.loc[not_numbers, col]
data.loc[105, col] = 1000
data[col] = data[col].astype(float)

            
            
'''
      ********** GASTOS DE SERVICIOS PUBLICOS DEL RESPONSABLE ***********
'''       
col = 'Gastos_SP_Gas'
data[col] = data[col].replace('bs','')
not_numbers = data[col].apply(is_not_number)
data.loc[not_numbers, col]
data.loc[105, col] = 2000
data[col] = data[col].astype(float)
data.info()
     
'''
      ********** GASTOS TOTALES DEL RESPONSABLE ***********
'''       
col = 'Total_Egresos_Resp'
data[col] = data[col].replace('bs','')
not_numbers = data[col].apply(is_not_number)
data.loc[not_numbers, col]
data.loc[32, col] = 24605.01
data.loc[86, col] = 10600
data.loc[105, col] = 20000
data.loc[133, col] = 16043.76
data.loc[142, col] = 54455.43
data[col] = data[col].astype(float)
data.info()

'''
      ********** GASTOS DE ALIMENTACION DEL RESPONSABLE ***********
'''    
     
col = 'Gastos_Alimentacion_Resp'
not_numbers = data[col].apply(is_not_number)
data.loc[not_numbers, col]
data.loc[105, col] = 7000
data[col] = data[col].astype(float)

'''
      ********** GASTOS DE ALIMENTACION DEL RESPONSABLE ***********
'''    

col = 'Gastos_Transporte_Resp'
data[col].replace('bs', '')
data.loc[105, col] = 2000
data[col] = data[col].astype(float)

col = ['Vivienda', 'Gastos_Alimentacion_Resp', 'Gastos_Transporte_Resp',
       'Gastos_Medicos_Resp','Gastos_Odonto_Resp', 'Gastos_Educativos_Resp', 
       'Gastos_SP_Gas', 'Condominio_Resp', 'Otros_Gastos_Resp', 'Total_Egresos_Resp']

#data[col].replace('bs','').astype(float)
data[col].astype(float)
data.info()
data['N_Periodo'] = np.where(data['N_Periodo']=='I', 1, 2)
fecha_nac = data['F_Nacimiento'].str.split('/')
data['Dia_Nacimiento'] = 0
data['Mes_Nacimiento'] = 0
data['Anho_Nacimiento'] = 0
for i in range(190):
    for j in range(3):
        if j == 0:
            data.loc[i, 'Dia_Nacimiento'] = fecha_nac[i][j]
        elif j == 1:
            data.loc[i, 'Mes_Nacimiento'] = fecha_nac[i][j]
        else:
            data.loc[i, 'Anho_Nacimiento'] = fecha_nac[i][j]
    
col = ['Dia_Nacimiento', 'Mes_Nacimiento', 'Anho_Nacimiento']     
data[col] = data[col].astype(int)
data[col]  
#del data['Unnamed: 0.1']
#del data['Unnamed: 0']       
#del data['Tipo_Frec_Actividad']   
#del data['F_Motiv_Sol_Benef']   
#del data['Direccion_Res_Alq']   
#del data['Motivo_C']   
#del data['Reprobacion_Motiv']
del data['Periodo']      
#del data['Sugerencia_Recomendaciones']
del data['F_Nacimiento']
del data['Edad']


col = ['Vivienda', 'Gastos_Alimentacion_Resp', 'Gastos_Transporte_Resp',
       'Gastos_Medicos_Resp','Gastos_Odonto_Resp', 'Gastos_Educativos_Resp', 
       'Gastos_SP_Gas', 'Condominio_Resp', 'Otros_Gastos_Resp', 'Total_Egresos_Resp']
data[col] = data[col].astype(float)

col = ['E_Civil', 'Sexo', 'Escuela', 'Mod_Ingreso', 'A_Periodo', 'Matrimonio',
       'Tesis', 'Veces_Tesis', 'M_Aprobadas', 'C_Direccion', 'Otro_Beneficio']
data[col] = data[col].astype(int)

col = ['Ingreso_Responsable', 'Ingreso_Resp_Otros', 'Total_Ingreso_Resp']
data[col] = data[col].astype(float)
            
            
'''
    En este punto ya tenemos un data set totalmente numerico. De esa manera se
    cumple con el paso referenciado en el libro Data Mining: The Textbook, 
    llamado Data Portability
'''

'''
    ********** DATA CLEANING **********
'''
data[['Monto_Alquiler', 'Gastos_Alquiler']]
data['Gastos_Alquiler'] = np.where( data['Gastos_Alquiler'] >= data['Monto_Alquiler'], \
                                    data['Gastos_Alquiler'], data['Monto_Alquiler'])
del data['Monto_Alquiler']
del fcol[2]
fcol.remove('Monto_Alquiler')
col = 'Monto_Mensual_Beca'
data.loc[data[col] > 1500, col]
data.loc[140, col] = 1500
ingresos = ['Monto_Mensual_Beca', 'Mensual_Aporte_Resp',
            'Mensual_Aporte_Amigos', 'Aporte_Destajo']
gastos = ['Alimentacion', 'T_Publico', 'Gastos_Medicos',
         'Gastos_Odonto', 'Gastos_Personales', 'Gastos_Alquiler','Gastos_Materiales',
         'Gastos_Recreacion', 'Gastos_Otros']
sum_gastos = data[gastos].sum(axis=1)
sum_gastos == data['Total_Egresos']
data['Total_Egresos'] = np.where( sum_gastos > data['Total_Egresos'], 
                                  sum_gastos , 
                                  data['Total_Egresos'])            
sum_ingresos = data[ingresos].sum(axis=1)
data['Mensual_Ingreso_Total'] == sum_ingresos
data['Mensual_Ingreso_Total'] = np.where( sum_ingresos > data['Mensual_Ingreso_Total'], 
                                          sum_ingresos , 
                                          data['Mensual_Ingreso_Total'])            
ingresos_resp = ['Ingreso_Responsable', 'Ingreso_Resp_Otros']
sum_ingresos_resp = data[ingresos_resp].sum(axis=1)
data['Total_Ingreso_Resp'] == sum_ingresos_resp
data['Total_Ingreso_Resp'] = np.where( data['Total_Ingreso_Resp'] > sum_ingresos_resp, 
                                        data['Total_Ingreso_Resp'], 
                                        sum_ingresos_resp)                                
gastos_resp = ['Vivienda', 'Gastos_Alimentacion_Resp', 'Gastos_Transporte_Resp','Gastos_Medicos_Resp', 
         'Gastos_Odonto_Resp', 'Gastos_Educativos_Resp', 'Gastos_SP_Gas', 'Condominio_Resp', 
         'Otros_Gastos_Resp']
sum_gastos_resp = data[gastos_resp].sum(axis=1)
sum_gastos_resp == data['Total_Egresos_Resp']
data['Total_Egresos_Resp'] = np.where( data['Total_Egresos_Resp'] < sum_gastos_resp,
                                        sum_gastos_resp,
                                        data['Total_Egresos_Resp'])                                        
data.to_csv('./minable.csv',
            sep=',')                               
'''
     ********** ANALISIS EXPLORATORIO DE LOS DATOS **********
'''
stats = data[fcol].describe()
stats_ingresos = data[ingresos].describe()
stats_gastos = data[gastos].describe()
stats_ingresos_total = data['Mensual_Ingreso_Total'].describe()
stats_gastos_total = data['Total_Egresos'].describe()
stats_ingresos_resp = data[ingresos_resp].describe()
stast_gastos_resp = data[gastos_resp].describe()
stats_ingreso_total_resp = data['Total_Ingreso_Resp'].describe()
stats_ereso_total_resp = data['Total_Egresos_Resp'].describe()

plt.figure()
data[ingresos].hist(figsize=(6,4))
data[gastos].hist()
data['Mensual_Ingreso_Total'].plot(kind='kde')
data['Total_Egresos'].plot(kind='kde')
data['Mensual_Ingreso_Total'].plot(kind='kde')
data[['Mensual_Ingreso_Total', 'Total_Egresos']].hist()

    
'''
    ********** DIMENSIONALITY REDUCTION **********
'''
#del data['Unnamed: 0']
'''
pca = PCA(n_components=2)
data_r = pca.fit(data_scaled).transform(data_scaled.as_matrix())
print('Varianza explicada en los 2 primeros componentes: %s'
      % str(pca.explained_variance_ratio_))

idata = data[icol]
bernoulli = ['Sexo', 'Matrimonio', 'C_Direccion', 'Tesis', 'Otro_Beneficio', 
             'A_Ingreso']
numpy_data = data[bernoulli].as_matrix()
for col in bernoulli:
    if data[col].var() < 0.016:
        del data[col]
  
data = pd.read_csv('./minable.csv', 
                   sep=',')
'''                   