"""
Libreria para interactuar con las API de FCV Central
"""


import datetime
import pprint
import requests
import pandas as pd


class ApiFcvEjemplo:
    """
    ApiFcvEjemplo
    """

    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint

    def _test(self):
        # endpoint = "https://api.publicapis.org/entries"
        # endpoint = "https://jsonplaceholder.typicode.com/todos"
        endpoint = "https://jsonplaceholder.typicode.com/users"

        resource_id = 2

        print("GET")
        # api_url = f"{api_endpoint}/{resource_id}"
        api_url = f"{endpoint}"
        response = requests.get(api_url, timeout=60)
        res = response.json()
        print(f"json {res}")
        res = response.headers["Content-Type"]
        print(f"type {res}")
        res = response.status_code
        print(f"code {res}")

        print("POST")
        api_url = f"{endpoint}"
        jdict = {
            "userId": resource_id,
            "title": "Buy milk",
            "completed": False,
        }
        response = requests.post(api_url, json=jdict, timeout=60)
        res = response.json()
        print(f"json {res}")
        res = response.headers["Content-Type"]
        print(f"type {res}")
        res = response.status_code
        print(f"code {res}")

        print("PUT")
        api_url = f"{endpoint}/{resource_id}"
        jdict = {
            "userId": resource_id,
            "title": "Buy milk",
            "completed": True,
            # "id": resource_id,
        }
        response = requests.put(api_url, json=jdict, timeout=60)
        res = response.json()
        print(f"json {res}")
        res = response.headers["Content-Type"]
        print(f"type {res}")
        res = response.status_code
        print(f"code {res}")

        print("DELETE")
        api_url = f"{endpoint}/{resource_id}"
        response = requests.delete(api_url, timeout=60)
        res = response.json()
        print(f"json {res}")
        res = response.headers["Content-Type"]
        print(f"type {res}")
        res = response.status_code
        print(f"code {res}")

        print("GET")
        api_url = f"{endpoint}/{resource_id}"
        response = requests.get(api_url, timeout=60)
        res = response.json()
        print(f"json {res}")
        res = response.headers["Content-Type"]
        print(f"type {res}")
        res = response.status_code
        print(f"code {res}")


class ApiFcvPlanograma:
    """
    ApiFcvPlanograma
    """

    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint

    def execute_request(self, local: int):
        # def get_planograma(self, id_trx: int, sku_arr: list, local: int):
        """
        get_planograma
        """
        # INPUT
        jdict = {
            # "Origen": "DeepView",
            # "Password": "abcd123",
            # "Contexto": "Quiebres",
            # "IdTransaccion": id_trx,
            # "Local": local,
            # "ListaSKU": sku_arr
        }

        t0 = datetime.datetime.now()

        # http://10.193.28.6:7001/MasGestion/resources/mg/quiebres/planograma/data/local/3/categoria/ACCESORIOS%20INFANTILES/modulacion/1G-PREM-BAJO
        api_url = f"{self.endpoint}/MasGestion/resources/mg/quiebres/planograma/data/local/{local}"  # Ok
        # api_url = f"{self.endpoint}/MasGestion/resources/mg/producto/stock/origen/Deep" # Error
        response = requests.get(api_url, json=jdict, timeout=60)
        # api_url = f"{self.endpoint}/MasGestion/resources/mg/quiebres/jda/local/{local}"
        # response = requests.post(api_url, json=jdict)
        jdict = response.json()

        t1 = datetime.datetime.now()
        dt = t1 - t0

        jdict["dt_ms"] = dt.microseconds / 1000

        return jdict

    def check_request(self, jdict: dict):
        """
        check_jdict
        """
        print(f"Verificando respuesta para {self.__class__.__name__}")

        dt_ms = jdict["dt_ms"]
        print(f"  Delta time {dt_ms} ms")

        # print(jdict)
        # pprint.pprint(jdict)
        print(f"  Campos recibidos {list(jdict.keys())}")
        data = jdict["data"]
        data0_keys = jdict["data"][0].keys()
        print(f"  Número de elementos en data {len(data)}")
        print(f"  Campos detectados en data[0] {list(data0_keys)}")

        # var = jdict["data"][0]
        # pprint.pprint(var)
        # pprint.pprint(var.keys())

        # Resultados prueba 28 de Febrero
        # asd = [
        #     'exhibicion',                     EX_TOTAL
        #     'apilar',                         APILAR
        #     'cod_interno',                    CODIGO
        #     'ctgr_nombre': 'MAQUILLAJE',      CATEGORIA
        #     'id_modulacion': '1MMPB-4',       ID MODULACION
        #     'locl_fecha_inicio_vigencia',     LIVEDATE
        #     'posicion',                       POSICION
        #     'caras',                          CARAS
        #     'upc',                            UPC
        #     'nombre_producto',                DESCRIPTOR
        #     'bandeja'                         ID_MOBILIARIO
        # ]


class ApiFcvStock:
    """
    ApiFcvStock
    """

    def __init__(self, ip_port) -> None:
        self.ip_port = ip_port

    def check_request(self, jdict):
        """
        check_stock_response
        """
        # Resultados prueba 28 de Febrero
        # OUTPUT OK
        # {'data': [{'contexto': 'Quiebres',
        #         'idTransaccion': 100500,
        #         'listaSKU': [{'SKU': '269467',
        #                         'fechaActualizacionSC': '2019-10-26 10:54:38.0',
        #                         'fechaActualizacionSL': '',
        #                         'stockCentral': 290,
        #                         'stockLocal': -999},
        #                         {'SKU': '269468',
        #                         'fechaActualizacionSC': '2019-10-26 10:54:38.0',
        #                         'fechaActualizacionSL': '',
        #                         'stockCentral': 290,
        #                         'stockLocal': -999},
        #                         {'SKU': '269469',
        #                         'fechaActualizacionSC': '2019-10-26 10:54:38.0',
        #                         'fechaActualizacionSL': '',
        #                         'stockCentral': 290,
        #                         'stockLocal': -999}],
        #         'local': 15,
        #         'origen': 'DeepView'}],
        # 'message': 'Consulta realizada satisfactoreamente.',
        # 'success': True}

        # OUTPUT ERROR
        # {'code': 10001, 'message': 'Origen o password incorrecto', 'success': False}

        assert isinstance(jdict, dict)

        print(f"Verificando respuesta para {self.__class__.__name__}")

        dt_ms = jdict["dt_ms"]
        print(f"  Delta time {dt_ms} ms")

        print(f"  Campos recibidos {list(jdict.keys())}")
        try:
            data = jdict["data"]
            data0_keys = jdict["data"][0].keys()
        except KeyError:
            pass
        else:
            print(f"  Número de elementos en data {len(data)}")
            print(f"  Campos detectados en data[0] {list(data0_keys)}")

        for key, val in jdict.items():
            # print(f"key {key}, val {val}")
            # print(f"key {key}")
            # continue

            target_key = "success"
            if key == target_key:
                print(f"  Se detecta el campo {target_key} con el estado {val}")
                if val:
                    print("  La respuesta es valida")
                else:
                    print("  La respuesta NO es valida")

            target_key = "message"
            if key == target_key:
                print(f"  Se detecta el campo {target_key} con el mensaje {val}")

            target_key = "data"
            if key == target_key:
                print(
                    f"  Se detecta el campo {target_key} con las llaves {list(val[0].keys())}"
                )
                lsku = val[0]["listaSKU"]
                print(
                    f"  Se detecta el campo {target_key}, la listaSKU tiene un largo {len(lsku)}"
                )
                print(
                    f"  Los campos de cada elemento de la listaSKU son {list(lsku[0].keys())}"
                )
                # print(f"Se detecta el campo {target_key}: {val}")

            target_key = "code"
            if key == target_key:
                print(f"  Se detecta el campo {target_key} con el código {val}")

            target_key = "error"
            if key == target_key:
                print(
                    f"  Se detecta el campo {target_key} con una lista de {len(val['errors'])} errores"
                )

    def execute_request(self, id_trx: int, sku_arr: list, local: int):
        """
        get_stock
        """
        # INPUT
        jdict = {
            "Origen": "ANDAIN",
            "Password": "ABCDEJ",
            "Contexto": "Quiebres",
            "IdTransaccion": id_trx,
            "Local": local,
            "ListaSKU": sku_arr,
        }

        # jdict = {
        #     "Origen": "ANDAIN",
        #     "Password": "ABCDEJ",
        #     "Contexto": "Quiebres",
        #     "IdTransaccion": 987,
        #     "Local": 15,
        #     "ListaSKU": [{"SKU": "269469"}, {"SKU": "269468"}, {"SKU": "269467"}],
        # }

        t0 = datetime.datetime.now()

        # http://10.193.28.6:7001/MasGestion/resources/mg/producto/stock
        api_endpoint = f"{self.ip_port}/MasGestion/resources/mg/producto/stock/"  # Ok
        # api_url = f"{self.endpoint}/MasGestion/resources/mg/producto/stock/origen/Deep" # Error
        response = requests.post(api_endpoint, json=jdict, timeout=60)
        # api_url = f"{self.endpoint}/MasGestion/resources/mg/quiebres/jda/local/{local}"
        # response = requests.post(api_url, json=jdict)
        jdict = response.json()

        t1 = datetime.datetime.now()
        dt = t1 - t0

        jdict["dt_ms"] = dt.microseconds / 1000

        return jdict


class ApiFcvResmed:
    """
    ApiFcvResauditoria
    """

    def __init__(self, ip_port) -> None:
        self.ip_port = ip_port

    def check_request(self, jdict):
        """
        check_jdict
        """

        assert isinstance(jdict, dict)

        print(f"Verificando respuesta para {self.__class__.__name__}")

        dt_ms = jdict["dt_ms"]
        print(f"  Delta tiempo {dt_ms} ms")

        print(f"  Campos recibidos {list(jdict.keys())}")
        try:
            data = jdict["data"]
            data0_keys = jdict["data"][0].keys()
        except KeyError:
            pass
        else:
            print(f"  Número de elementos en data {len(data)}")
            print(f"  Campos detectados en data[0] {list(data0_keys)}")

    def execute_request(
        self, id_operation: int, type_operation: int, result_arr: list, ver: str
    ):
        """
        get_stock
        """
        # INPUT
        jdict = {
            "idOperacion": id_operation,
            "TipoOperacion": type_operation,
            "ListaResultados": result_arr,
            "ver": ver,
        }

        t0 = datetime.datetime.now()

        # http://10.193.28.6:7001/MasGestion/resources/mg/producto/stock
        api_endpoint = f"{self.ip_port}/MasGestion/resources/mg/medicion/guardar"
        response = requests.post(api_endpoint, json=jdict, timeout=60)
        jdict = response.json()

        t1 = datetime.datetime.now()
        dt = t1 - t0

        jdict["dt_ms"] = dt.microseconds / 1000

        return jdict


if __name__ == "__main__":
    print("")
    print("-----------------------------------------------------------------")
    print("Antes de ejecutar este código se debe tener la VPN conectada.")
    print("Se deben seguir los pasos descritos en data/datos_vpn_socofar.txt")
    print("-----------------------------------------------------------------")
    print("")

    API_IP_PORT = "http://10.193.28.6:7001"

    apiplano = ApiFcvPlanograma(API_IP_PORT)
    NUM_LOCAL = 3
    # num_local = "FCV0026"
    # num_local = 531101
    jd = apiplano.execute_request(NUM_LOCAL)
    # apiplano.check_request(jd)
    # jd a df
    df = pd.DataFrame(jd["data"])
    #save csv
    df.to_csv(f"planograma_{NUM_LOCAL}.csv", index=False)
    # apistock = ApiFcvStock(API_IP_PORT)
    # ID_TRX_ANDAIN = 123
    # # requested_sku = [{"SKU": "555"}, {"SKU": "8266"}]
    # requested_sku = [{"SKU": "269469"}, {"SKU": "269468"}, {"SKU": "269467"}]
    # NUM_LOCAL = 26
    # jd = apistock.execute_request(ID_TRX_ANDAIN, requested_sku, NUM_LOCAL)
    # apistock.check_request(jd)

    # apistock = ApiFcvResmed(API_IP_PORT)
    # ID_OPERATION = 123123
    # TYPE_OPERATION = 456456
    # result_arr = [
    #     {
    #         "id": 1378,
    #         "user": "Local-0400-auditor-1",
    #         "local": 400,
    #         "categoria": "AFEITADO",
    #         "id_modulacion": "1G-ALTOMEDIO",
    #         "sku": 287968,
    #         "descriptor": "GILLE.CAJA FICTICIOS:1G-S",
    #         "id_mobiliario": 1,
    #         "posicion": 1,
    #         "caras": 1,
    #         "apilar": 1,
    #         "profundidad": 1,
    #         "ex_total": 1,
    #         "livedate": "7/26/2022 0:00",
    #         "inicio_auditoria": "7/27/2022 10:55",
    #         "termino_auditoria": "7/27/2022 11:10",
    #         "inicio_procesamiento": "7/27/2022 10:57",
    #         "termino_procesamiento": "7/27/2022 10:58",
    #         "unidades_totales": 1,
    #         "unidades_identificadas_rev1": 0,
    #         "unidades_pistoleadas": 0,
    #         "foto_banco": 0,
    #         "alarma_desactualizada": 0,
    #         "stock": 0,
    #         "status": "OK",
    #         "imagenes": "imagenes id=1378",
    #     },
    #     {
    #         "id": 1378,
    #         "user": "Local-0400-auditor-1",
    #         "local": 400,
    #         "categoria": "AFEITADO",
    #         "id_modulacion": "1G-ALTOMEDIO",
    #         "sku": 402106,
    #         "descriptor": "GROOMEN 300 MAQAF+3CART",
    #         "id_mobiliario": 2,
    #         "posicion": 1,
    #         "caras": 1,
    #         "apilar": 1,
    #         "profundidad": 1,
    #         "ex_total": 1,
    #         "livedate": "7/26/2022 0:00",
    #         "inicio_auditoria": "7/27/2022 10:55",
    #         "termino_auditoria": "7/27/2022 11:10",
    #         "inicio_procesamiento": "7/27/2022 10:57",
    #         "termino_procesamiento": "7/27/2022 10:58",
    #         "unidades_totales": 1,
    #         "unidades_identificadas_rev1": 2,
    #         "unidades_pistoleadas": 0,
    #         "foto_banco": 1,
    #         "alarma_desactualizada": 0,
    #         "stock": 0,
    #         "status": "OK",
    #         "imagenes": "imagenes id=1378",
    #     },
    # ]
    # VER = "423a16437c11a8db8d00ad99d8b9e415"
    # jd = apistock.execute_request(ID_OPERATION, TYPE_OPERATION, result_arr, VER)
    # apistock.check_request(jd)
