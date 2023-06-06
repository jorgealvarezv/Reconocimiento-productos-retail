#code of crud of bd in postgresql with python in the squema "cruzverde" and table maestro,planogramas,locales,id_modulacion,banco_fotos and Stock in localhost port 5432 with user postgres and password postgres
# -*- coding: utf-8 -*-
import csv
import shutil
import psycopg
import pandas as pd
import numpy as np
import ast
import api_central as api
import os

#conexion a la base de datos
def conexion(dbname='deepview01', user='postgres', host='localhost', password='postgres', port='5432'):
    try:
        conn = psycopg.connect(dbname=dbname, user=user, host=host, password=password, port=port)
        cur = conn.cursor()
        return conn,cur
    except:
        print("unable to connect to the database")

def closeConexion(conexion,cursor):
    conexion.close()
    cursor.close()
# 
# #funcion para crear un nuevo registro en la tabla maestro
def insertSku(conexion,cursor,sku):
    cursor.execute("""SELECT sku FROM cruzverde.skus WHERE sku = %s""", (str(sku),))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO cruzverde.skus (sku) VALUES (%s)""", (int(sku),))
        conexion.commit()

def removeSku(conexion,cursor,sku):
    cursor.execute("""DELETE FROM cruzverde.skus WHERE sku = %s""", (str(sku),))
    conexion.commit()



def createSkus(conexion,cursor,xlxs):
    # if maestro table not exist create it

    df=pd.read_excel(xlxs)

    # Eliminar duplicados y columnas no deseadas
    df.drop_duplicates(subset=['sku'], keep='first', inplace=True)
    df.drop(['descriptor','upc'], axis=1, inplace=True)
    
    # Crear la consulta SQL
    insert_stmt = "INSERT INTO cruzverde.skus (sku) VALUES (%s)"

    # Convertir los datos a una lista de tuplas
    tuples = [tuple(x) for x in df.values]


    # Convertir los valores numéricos a un tipo compatible con PostgreSQL
    tuples = [(int(x[0]),) for x in tuples]
    
    cursor.execute("""SELECT * FROM cruzverde.skus""")
    if cursor.rowcount == 0:
        cursor.executemany(insert_stmt, tuples)
        conexion.commit()
    else:
        #compare the df with the table and update the table efficiently
        for index, row in df.iterrows():
            cursor.execute("""SELECT * FROM cruzverde.skus WHERE sku = %s""", (str(row['sku']),))
            if cursor.rowcount == 0:
                insertSku(conexion,cursor,row['sku'])

def insertMaestro(conexion,cursor,sku,descriptor,upc):
    if len(str(upc)) < 20:
        # un upc solo puede tener un unico sku asi que si ya hay un sku con ese upc no se puede ingresar:
        cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
        if cursor.rowcount != 0:
            sku_id = cursor.fetchone()[0]

            cursor.execute("""SELECT sku_id FROM cruzverde.maestro WHERE upc = %s""", (str(upc),))
            if cursor.rowcount == 0:
                cursor.execute("""INSERT INTO cruzverde.maestro (sku_id,descriptor,upc) VALUES (%s,%s,%s)""", (int(sku_id), str(descriptor), str(upc)))
                conexion.commit()
                #print(f'Se ingreso el registro correctamente: sku:{sku}, descriptor:{descriptor}, upc:{upc}')
            elif cursor.rowcount >0:
                print(f'El upc:{upc} ya existe para otro sku')
            else:
                print('error')

def updateMaestro(conexion,cursor,sku,descriptor,upc):
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    if len(str(upc)) < 20:
        cursor.execute("""UPDATE cruzverde.maestro SET sku_id = %s, descriptor = %s WHERE upc = %s""", (int(sku_id), str(descriptor), str(upc)))
        conexion.commit()

def removeMaestro(conexion,cursor,upc):
    
    if len(str(upc)) < 20:
        #drop cascade by upc:
        cursor.execute("""DELETE FROM cruzverde.maestro WHERE upc = %s""", (str(upc),))
        conexion.commit()
def createMaestro(conexion,cursor,xlsx):
    print("Actualizando tabla Skus")
    createSkus(conexion,cursor,xlsx)
    print("Tabla skus actualizada")

    df = pd.read_excel(xlsx)
    df=df.drop_duplicates()
    
    cursor.execute("""SELECT * FROM cruzverde.maestro""")
    if cursor.rowcount == 0:
        print("Tabla maestro vacia, creando tabla maestro")
        for index, row in df.iterrows():
            insertMaestro(conexion,cursor,row['sku'],row['descriptor'],str(int(row['upc'])))
    else: 
        print("Tabla maestro no vacia, actualizando tabla maestro")
        for index, row in df.iterrows():
            upc=str(int(row['upc']))
            cursor.execute("""SELECT * FROM cruzverde.maestro WHERE upc = %s""", (str(upc),))
            #inner join to get the sku_id
            
            if cursor.rowcount == 0:
                #print(f'El upc:{upc} no existe, creando registro')
                insertMaestro(conexion,cursor,row['sku'],row['descriptor'],str(int(row['upc'])))
            elif cursor.rowcount > 0:
                updateMaestro(conexion,cursor,row['sku'],row['descriptor'],str(int(row['upc'])))
        cursor.execute("""SELECT * FROM cruzverde.maestro""")
        for row in cursor.fetchall():
            if row[2] not in df['upc'].values:
                removeMaestro(conexion,cursor,row[2])
    
def insert_ocr(conexion,cursor,sku,words):
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    #solo en memoria cambiar en productivo
    if cursor.rowcount == 0:
        insertSku(conexion,cursor,sku)
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    cursor.execute("""INSERT INTO cruzverde.ocr (sku,words) VALUES (%s,%s)""", (int(sku_id), str(words)))
    conexion.commit()
def update_ocr(conexion,cursor,sku,words):
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    cursor.execute("""UPDATE cruzverde.ocr SET words = %s WHERE sku = %s""", (str(words), int(sku_id)))
    conexion.commit()
def remove_ocr(conexion,cursor,sku):
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    cursor.execute("""DELETE FROM cruzverde.ocr WHERE sku = %s""", (int(sku_id),))
    conexion.commit()
def create_ocr(conexion,cursor,csv):
    df = pd.read_csv(csv)
    df=df.drop_duplicates()
    #drop duplicates only consideren the sku column:
    df=df.drop_duplicates(subset=['SKU'])
    df=df.dropna()
    cursor.execute("""SELECT * FROM cruzverde.ocr""")
    if cursor.rowcount == 0:
        print("Tabla ocr vacia, poblando tabla ocr")
        for index, row in df.iterrows():
            insert_ocr(conexion,cursor,row['SKU'],row['Text'])
    else: 
        print("Tabla ocr no vacia, actualizando tabla ocr")
        for index, row in df.iterrows():
            #cursor.execute("""SELECT * FROM cruzverde.ocr WHERE sku = %s""", (int(row['SKU']),))
            #inner join sku_id with skus table
            cursor.execute("""SELECT ocr.sku FROM cruzverde.ocr INNER JOIN cruzverde.skus ON ocr.sku = skus.id WHERE skus.sku = %s""", (int(row['SKU']),))
            if cursor.rowcount == 0:
                #print(f'El sku: {row["SKU"]} no existe, creando registro')
                #print(row['SKU'],row['Text'])
                insert_ocr(conexion,cursor,row['SKU'],row['Text'])
            elif cursor.rowcount > 0:
                update_ocr(conexion,cursor,row['SKU'],row['Text'])
        #remove ocr that are not in the csv inner join sku_id with skus table
        cursor.execute("""SELECT skus.sku FROM cruzverde.ocr INNER JOIN cruzverde.skus ON ocr.sku = skus.id""")
        for row in cursor.fetchall():
            if row[0] not in df['SKU'].values:
                remove_ocr(conexion,cursor,row[0])

def insertClip(conexion,cursor,sku,description):
    #inner join sku_id with skus table
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    if(cursor.rowcount == 0):
        insertSku(conexion,cursor,sku)
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    cursor.execute("""INSERT INTO cruzverde.clip (sku,descripcion) VALUES (%s,%s)""", (int(sku_id), str(description)))
    conexion.commit()
def updateClip(conexion,cursor,sku,description):
    #inner join sku_id with skus table
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    cursor.execute("""UPDATE cruzverde.clip SET descripcion = %s WHERE sku = %s""", (str(description), int(sku_id)))
    conexion.commit()
def removeClip(conexion,cursor,sku):
    #inner join sku_id with skus table
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    sku_id = cursor.fetchone()[0]
    cursor.execute("""DELETE FROM cruzverde.clip WHERE sku = %s""", (int(sku_id),))
    conexion.commit()
def createClip(conexion,cursor,xlsx):
    df = pd.read_excel(xlsx)
    df=df.drop_duplicates()
    #drop duplicates only consideren the sku column:
    df=df.drop_duplicates(subset=['Sku'])
    df=df.dropna()
    cursor.execute("""SELECT * FROM cruzverde.clip""")
    if cursor.rowcount == 0:
        print("Tabla clip vacia, poblando tabla clip")
        for index, row in df.iterrows():
            insertClip(conexion,cursor,row['Sku'],row['Descripcion'])
    else: 
        print("Tabla clip no vacia, actualizando tabla clip")
        for index, row in df.iterrows():
            #cursor.execute("""SELECT * FROM cruzverde.clip WHERE sku = %s""", (int(row['SKU']),))
            #inner join sku_id with skus table
            cursor.execute("""SELECT clip.sku FROM cruzverde.clip INNER JOIN cruzverde.skus ON clip.sku = skus.id WHERE skus.sku = %s""", (int(row['Sku']),))
            if cursor.rowcount == 0:
                #print(f'El sku: {row["SKU"]} no existe, creando registro')
                #print(row['SKU'],row['Text'])
                insertClip(conexion,cursor,row['Sku'],row['Descripcion'])
            elif cursor.rowcount > 0:
                updateClip(conexion,cursor,row['Sku'],row['Descripcion'])
        #remove clip that are not in the csv inner join sku_id with skus table
        cursor.execute("""SELECT skus.sku FROM cruzverde.clip INNER JOIN cruzverde.skus ON clip.sku = skus.id""")
        for row in cursor.fetchall():
            if row[0] not in df['Sku'].values:
                removeClip(conexion,cursor,row[0])
def insertLocales(conexion,cursor,codigo_local):
    local='FCV'+str(codigo_local).zfill(4)
    cursor.execute("""SELECT * FROM cruzverde.locales WHERE codigo_local = %s""", (codigo_local,))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO cruzverde.locales (codigo_local,local) VALUES (%s,%s)""", (codigo_local, local))
        conexion.commit()
def updateLocales(conexion,cursor,codigo_local):
    local='FCV'+str(codigo_local).zfill(4)
    cursor.execute("""UPDATE cruzverde.locales SET local = %s WHERE codigo_local = %s""", (local, codigo_local))
    conexion.commit()  
def removeLocales(conexion,cursor,codigo_local):
    cursor.execute("""DELETE FROM cruzverde.locales WHERE codigo_local = %s""", (codigo_local,))
    conexion.commit()
# planograma a df 

def insertId_Modulacion(conexion,cursor,id_modulacion):
    cursor.execute("""SELECT * FROM cruzverde.id_modulacion WHERE id_modulacion = %s""", (id_modulacion,))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO cruzverde.id_modulacion (id_modulacion) VALUES (%s)""", (id_modulacion,))
        conexion.commit()
def updateId_Modulacion(conexion,cursor,id_modulacion):
    cursor.execute("""UPDATE cruzverde.id_modulacion SET id_modulacion = %s""", (id_modulacion,))
    conexion.commit()
def removeId_Modulacion(conexion,cursor,id_modulacion):
    cursor.execute("""DELETE FROM cruzverde.id_modulacion WHERE id_modulacion = %s""", (id_modulacion,))
    conexion.commit()
    

def insertPlanograma(conexion,cursor,datos):
    local,categoria,id_modulacion,posicion,sku_id,id_mobiliario,caras,apilar,profundidad,ex_total,livedate=datos
    #primary is local,id_modulacion,posicion,id_mobiliario:
    cursor.execute("""SELECT * FROM cruzverde.planogramas WHERE local = %s AND categoria=%s AND id_modulacion = %s AND posicion = %s AND id_mobiliario = %s""", (local,categoria,id_modulacion,posicion,id_mobiliario))
    if cursor.rowcount == 0:
        cursor.execute("""INSERT INTO cruzverde.planogramas (local,categoria,id_modulacion,posicion,sku_id,id_mobiliario,caras,apilar,profundidad,ex_total,livedate) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (local,categoria,id_modulacion,posicion,sku_id,id_mobiliario,caras,apilar,profundidad,ex_total,livedate))
        conexion.commit()
    else:
        print (f'El planograma {local},{id_modulacion},{posicion},{id_mobiliario} ya existe')
def updatePlanograma(conexion,cursor,datos):
    local,categoria,id_modulacion,posicion,sku_id,id_mobiliario,caras,apilar,profundidad,ex_total,livedate=datos
    #primary is local,id_modulacion,posicion,id_mobiliario:
    cursor.execute("""UPDATE cruzverde.planogramas SET categoria = %s, sku_id = %s, caras = %s, apilar = %s, profundidad = %s, ex_total = %s, livedate = %s WHERE local = %s AND id_modulacion = %s AND posicion = %s AND id_mobiliario = %s""", (categoria,sku_id,caras,apilar,profundidad,ex_total,livedate,local,id_modulacion,posicion,id_mobiliario))
    conexion.commit()
def removePlanograma(conexion,cursor,datos):
    local,categoria,id_modulacion,posicion,id_mobiliario=datos
    cursor.execute("""DELETE FROM cruzverde.planogramas WHERE local = %s AND categoria=%s AND id_modulacion = %s AND posicion = %s AND id_mobiliario = %s""", (local,categoria,id_modulacion,posicion,id_mobiliario))


    
    

def createPlanograma(conexion,cursor,local):
    planograma=conect_api_planograma(local=local)
    if(planograma.empty):
        print("No hay planograma para este local")
    else:
        #insertLocales(conexion,cursor,local)
        cursor.execute("""SELECT * FROM cruzverde.planogramas WHERE local = %s""", (int(local),))
        if cursor.rowcount != 0:
            print(f"Tabla planograma no vacia para el local {local}, borrando datos")
            cursor.execute("""SELECT local,categoria,id_modulacion,posicion,id_mobiliario FROM cruzverde.planogramas WHERE local = %s""", (local,))
            for row in cursor.fetchall():
                datos=(row[0],row[1],row[2],row[3],row[4])
                removePlanograma(conexion,cursor,datos)
            
        print(f"Insertando planograma local {local} datos desde la api")  
        insertLocales(conexion,cursor,local)
        for index, row in planograma.iterrows():
            #print(row)
            profundidad=1 #no se encuentra en la api
            sku= sku_id(cursor,int(row['cod_interno']))
            id_modulacion= row['id_modulacion']
            insertId_Modulacion(conexion,cursor,id_modulacion)
            cod_modulacion=codigo_modulacion(cursor,id_modulacion)
            if sku != 0:
                datos=(local,row['ctgr_nombre'],cod_modulacion,row['posicion'],sku,row['bandeja'],row['caras'],row['apilar'],profundidad,row['exhibicion'],row['locl_fecha_inicio_vigencia'])
                insertPlanograma(conexion,cursor,datos)
       
          
            
            

def id_to_sku(cursor,id):
    cursor.execute("""SELECT sku FROM cruzverde.skus WHERE id = %s""", (str(id),))
    if cursor.rowcount == 0:
        print("Error: No existe id")
        return 0
        
    return cursor.fetchone()[0]

def sku_id(cursor,sku):
    cursor.execute("""SELECT id FROM cruzverde.skus WHERE sku = %s""", (int(sku),))
    if cursor.rowcount == 0:
        print(f"Error: No existe sku {sku}")
        return 0
        
    return cursor.fetchone()[0]
def codigo_modulacion(cursor,id_modulacion):
    cursor.execute("""SELECT codigo_modulacion FROM cruzverde.id_modulacion WHERE id_modulacion = %s""", (str(id_modulacion),))
    if cursor.rowcount == 0:
        print("Error: No existe codigo de modulacion")
        return 0
        
    return cursor.fetchone()[0]
def get_id_modulacion(cursor,codigo_modulacion):
    cursor.execute("""SELECT id_modulacion FROM cruzverde.id_modulacion WHERE codigo_modulacion = %s""", (str(codigo_modulacion),))
    if cursor.rowcount == 0:
        print("Error: No existe codigo de modulacion")
        return 0
        
    return cursor.fetchone()[0]

def view_id_modulacion(conexion,cursor,local,categoria):
    cursor.execute("""SELECT id_modulacion FROM cruzverde.planogramas WHERE local = %s AND categoria = %s""", (local,categoria))

    
    return list(set([get_id_modulacion(cursor,item[0]) for item in cursor.fetchall()]))
    
def view_skus_gondolda(conexion,cursor,local,categoria,id_modulacion):
    codigo=codigo_modulacion(cursor,id_modulacion)
    cursor.execute("""SELECT skus.sku FROM cruzverde.planogramas INNER JOIN cruzverde.skus ON planogramas.sku_id = skus.id WHERE planogramas.local = %s AND planogramas.categoria = %s AND planogramas.id_modulacion = %s""", (local,categoria,codigo))
    return [item[0] for item in cursor.fetchall()]


def upcToSku(cursor,upc):
   #inner join sku_id with skus table
    cursor.execute("""SELECT skus.sku FROM cruzverde.maestro INNER JOIN cruzverde.skus ON maestro.sku_id = skus.id WHERE maestro.upc = %s""", (str(upc),))
    return cursor.fetchone()[0]
def skuToUpc(cursor,sku):
    cursor.execute("""SELECT maestro.upc FROM cruzverde.maestro INNER JOIN cruzverde.skus ON maestro.sku_id = skus.id WHERE skus.sku = %s""", (str(sku),))
    #return all upcs for a sku
    #to list
      
    return [item[0] for item in cursor.fetchall()]
def conect_api_planograma(port="http://10.193.28.6:7001",local=290):
    api_planograma=api.ApiFcvPlanograma(port)
    planograma=api_planograma.execute_request(local)
    df = pd.DataFrame(planograma["data"])
    df = df.drop_duplicates()
    return df

def wordsOfSku(cursor,sku):
    cursor.execute("""SELECT ocr.words FROM cruzverde.ocr INNER JOIN cruzverde.skus ON ocr.sku = skus.id WHERE skus.sku = %s""", (str(sku),))
    if(cursor.rowcount == 0):
        return []
    words=cursor.fetchall()[0][0]
    #literal_eval to convert string to list
    words=ast.literal_eval(words)
    return words
def descriptionOfSku(cursor,sku):
    cursor.execute("""SELECT clip.descripcion FROM cruzverde.clip INNER JOIN cruzverde.skus ON clip.sku = skus.id WHERE skus.sku = %s""", (str(sku),))
    if(cursor.rowcount != 0):
        return cursor.fetchone()[0]
    else:
        return ""
def categoryOfSku(cursor,sku):
    #category of sku in planograma and id_modulacion inner join with id_modulacion table
    cursor.execute("""SELECT planogramas.categoria,planogramas.id_modulacion FROM cruzverde.planogramas INNER JOIN cruzverde.skus ON planogramas.sku_id = skus.id WHERE skus.sku = %s""", (str(sku),))
    if(cursor.rowcount == 0):
        return []
    categories=set([ (cat,get_id_modulacion(cursor,id)) for cat, id in cursor.fetchall()])
        
    return list(categories)
# amount of skus in a category and id_modulacion with ocr and clip:
def skusInference(cursor,local,categoria,id_modulacion,folder):
    #get all skus in planograma
    skus=view_skus_gondolda(conexion,cursor,local,categoria,id_modulacion)
    #print(skus)
    skus_ocr=[]
    skus_clip=[]
    skus_folder=[]
    faltante_ocr=[]
    faltante_clip=[]
    for sku in skus:
        #see if sku has ocr

        if((os.path.exists(folder+str(sku).zfill(6))) != 0):
            skus_folder.append(sku)
    for sku in skus_folder:
        if(len(wordsOfSku(cursor,sku)) != 0):
            skus_ocr.append(sku)
        else:
            faltante_ocr.append(sku)
        #see if sku has clip
        if(len(descriptionOfSku(cursor,sku)) != 0):
            skus_clip.append(sku)
        else:
            faltante_clip.append(sku)
    #copia las carpetas de sku_folder a una carpeta dentro de img/testing/{id_modulacion}/{categoria} (creando las carpetas si no existen):
    # for sku in skus_folder:
    #     shutil.copytree(folder+str(sku).zfill(6), '/home/jorge/U/memoria-jorgealvarez/img/testing/'+str(id_modulacion)+'/'+categoria+'/'+str(sku).zfill(6))
    print(faltante_ocr)
    print(faltante_clip)
    
    return skus,skus_folder,skus_ocr ,skus_clip 
def wordsOfSkuToCsv(cursor,list_of_skus):
    #list of skus to csv
    with open('data/ocr_medicamentos.csv', 'w', newline='') as csvfile:
        fieldnames = ['SKU', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sku in list_of_skus:
            writer.writerow({'sku': sku, 'words': wordsOfSku(cursor,sku)})
def descriptionOfSkuToCsv(cursor,list_of_skus):
    #list of skus to csv
    with open('data/clip_medicamentos.csv', 'w', newline='') as csvfile:
        fieldnames = ['sku', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sku in list_of_skus:
            writer.writerow({'sku': sku, 'description': descriptionOfSku(cursor,sku)})



    
     

if __name__ == '__main__':
    conexion,cursor = conexion(dbname='cruzverde')
    # createMaestro(conexion,cursor,'/home/jorge/Deepview/CruzVerde/bases-de-datos-cruzverde/maestros/Codigo de barras_23_DIC.xlsx')
    # create_ocr(conexion,cursor,'/home/jorge/U/memoria-jorgealvarez/data/text_products.csv')
    # createClip(conexion,cursor,'/home/jorge/U/memoria-jorgealvarez/data/Descripciones_BD.xlsx')

    #print(desc)
    skus,skus_folder,skus_ocr,skus_clip=skusInference(cursor,3,'OAI FARMA', 'M10-OAI-F', '/home/jorge/U/memoria-jorgealvarez/img/bounding_boxes/')
    skus_clip.sort()
    skus_ocr.sort()
    wordsOfSkuToCsv(cursor,skus_ocr)
    descriptionOfSkuToCsv(cursor,skus_clip)
    #createPlanograma(conexion,cursor,3)
    closeConexion(conexion,cursor)
    
    





