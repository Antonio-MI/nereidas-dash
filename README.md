# nereidas-dash

Este repositorio contiene el código para reproducir el ejemplo de uso mostrado en el catálogo del proyecto NEREIDAS. 

Los datos utilizados en este visor proceden de la Universidad Politécnica de Cartagena (UPCT), en https://marmenor.upct.es/embed/l2, y del Instituto Murciano de Investigación y Desarrollo Agrario y Medioambiental (IMIDA) en https://idearm.imida.es/cgi/siomctdmarmenor/.

## `LecturaDatos.ipynb`

El notebook de `LecturaDatos.ipynb` hace lo siguiente:
- Carga los datos de las boyas, disponibles en el catálogo. Los datos para esto deben guardarse en la carpeta `BuoyData`, y después definir el path correspondiente en la primera celda del notebook.
- Cada uno de los csv asociado a los parámetros de Clorofila, Temperatura, Oxígeno Disuelto y Saliniidad se guardan en un diccionario de dataframes.
- Se itera sobre este diccionario para crear un único csv para cada uno de los parámetros.

## `dashapp.py`

Contiene el código documentado para generar el layout de la aplicación dash a partir de los csv creados con el notebook anterior.

