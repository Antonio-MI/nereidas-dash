# nereidas-dash

Este repositorio contiene el código para reproducir el ejemplo de uso mostrado en el catálogo del proyecto NEREIDAS.

## `LecturaDatos.ipynb`

El notebook de `LecturaDatos.ipynb` hace lo siguiente:
- Carga los datos de las boyas, disponibles en el catálogo.
- Cada uno de los csv asociado a los parámetros de Clorofila, Temperatura, Oxígeno Disuelto y Saliniidad se guardan en un diccionario de dataframes.
- Se itera sobre este diccionario para crear un único csv para cada uno de los parámetros.

## `dashapp.py`

Contiene el código documentado para generar el layout de la aplicación dash a partir de los csv creados con el notebook anterior.

