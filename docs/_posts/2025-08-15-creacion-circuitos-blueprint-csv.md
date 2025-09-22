---
layout: single
title:  "Creación de simulacion en Carla"
date:   2025-09-22 17:00:00 +0200
categories: jekyll update
---

# Introduccion

El primer paso en este TFG, tras la instalacion del simulador Carla, es la creacion de circuitos siguientdo las reglas de [Formula student Germany](https://www.formulastudent.de/fsg/rules), en las reglas podremos apreciar que hay 3 tipos distintos de circuitos, *skidpad*, *trackdrive*, *acceleration*. Estos circuitos seran creados usando conos, por lo que para facilitar la colocacion de estos he creado un **Blueprint** en Carla que coloca los conos automaticamente.

# Explicacion Blueprint

Para la colocacion de los conos de manera automatica siguiendo un recorrido he implemnetado un blueprint en Carla, este Blueprint consiste en dos funciones. La primera se encarga de colocar los conos siguiendo el recorrido marcado con un spline, este spline puede haber sido generado de manera automatica (importando un CSV con la posicion de los conos), o de manera manual. 

La segunda funcion convierte un CSV que contenga las coordenadas de los conos que se van a colocar, en un spline.

## Funcion de colocacion de conos

En la primera parte de la funcion se siguen los siguientes pasos:

1. Limpiar conos anteriores (`Clear Instances`): Lo primero que hace el blueprint es eliminar todas las instancias de conos amarillos (ISM Cono Amarillo) y azules (ISM Cono Azul) que pudieran existir. Esto asegura que cada vez que se ejecuta, el circuito se genera desde cero.

2. Comprobar si hay un CSV válido (`Is Valid`): El blueprint comprueba si una variable llamada Puntos DT (probablemente una Tabla de Datos o Data Table cargada desde un archivo CSV) es válida. Esto es para saber si se ha proporcionado un archivo con puntos para generar el trazado.

3. Decidir qué spline usar (`Branch`):
    * Si `Puntos DT` es válido (True): Significa que hay un CSV cargado. El blueprint establece la variable Spline A Usar para que sea Spline Csv 2 (el spline generado a partir del archivo).
    * Si `Puntos DT` no es válido (False): No hay un CSV. El blueprint usa un trazado por defecto, estableciendo Spline A Usar para que sea Path Spline (un spline dibujado manualmente en el editor).

![funcion col conos1]({{ site.baseurl }}/assets/images/funcion_col_conos1.jpeg)

Tras esto, la segunda parte de la funcion sigue los siguientes pasos:

1. Bucle `For Loop`: El blueprint comienza con un bucle que se repite para colocar cada uno de los conos de ese lado.

2. Calcular la distancia en el spline: En cada repetición del bucle, multiplica el Index por la separacion entre  conos. Esto calcula la distancia a lo largo del spline donde se debe colocar el siguiente cono.

3. Obtener la ubicación y dirección:
    * Usa Get Location at Distance Along Spline para obtener el punto exacto en el spline correspondiente a la distancia calculada.
    * Usa Get Right Vector at Distance Along Spline para obtener un vector que apunta hacia la "derecha" del spline en ese mismo punto.

4. Calcular la posición final del cono:
    * Multiplica el "vector derecho" por una la separacion entre  lineas para determinar qué tan lejos del spline se colocará el cono.
    * Suma este nuevo vector de desplazamiento a la ubicación original del spline. El resultado es la posición final del cono, a un lado del camino.

5. Crear el cono:
    * Get Rotation at Distance Along Spline obtiene la rotación del spline para orientar el cono.
    * Make Transform combina la posición final, la rotación y una escala por defecto (1,1,1).
    * Finalmente, Add Instance añade una nueva instancia del cono (en este caso, un ISMConoAmarill o Cono Amarillo) en la posición y rotación calculadas.

6. Repetir el mismo proceso con los conos del otro lado.

![funcion col conos2]({{ site.baseurl }}/assets/images/funcion_col_conos2.jpeg)

## Funcion para pasar de un CSV a un spline

En esta segunda funcion del Blueprint, se trata de obtener un spline a partir de un CSV. Para generar los CSVs he usado esta herramienta: https://github.com/mvanlobensels/random-track-generator

En el primer paso. El objetivo es leer los datos del archivo CSV y organizar las posiciones de los conos por color.

1. Inicio y Limpieza:
    * El proceso se inicia con el evento Csv To Spline.
    * Inmediatamente, vacía (Clear) los arrays Posiciones Azules y Posiciones Amarillas, y también borra todos los puntos
        (Clear Spline Points) del Spline Csv 2. Esto es para empezar de cero cada vez.
2. Lectura del Fichero (Data Table):
    * Se comprueba si la tabla de datos Puntos DT (cargada desde el CSV) es válida.
    * Si lo es, usa un bucle For Each Loop para recorrer cada una de las filas del fichero.
3. Clasificación por Color:
    * Dentro del bucle, para cada fila, rompe la estructura de datos (Break csv_point) para acceder a sus campos.
    * Se usa un Switch on String para comprobar el valor del campo Color.
    * Si el color es "yellow", añade la Location (posición) de ese cono al array Posiciones Amarillas.
    * Si el color es "blue", añade la Location al array Posiciones Azules.

![function csv 1]({{ site.baseurl }}/assets/images/function_csv1.jpeg)

En el segundo paso. Se usan las listas de conos creadas en el paso anterior para construir el trazado central del circuito.

1. Preparar el Bucle:
    * Se mide la longitud (Length) de las dos listas (Posiciones Azules y Posiciones Amarillas).
    * Se calcula el valor mínimo (MIN) entre las dos longitudes. Esto es una medida de seguridad para asegurarse de que solo
        procesa pares de conos (uno azul y uno amarillo).
    * Se usa este número para definir cuántas veces se ejecutará el For Loop.
2. Calcular el Punto Medio:
    * Dentro del bucle, para cada índice, se coge la posición del cono amarillo (GET de Posiciones Amarillas) y la del cono azul
        (GET de Posiciones Azules).
    * Suma (+) las dos posiciones y divide (/) el resultado por 2.0. Esto calcula el punto exacto que está en medio de los dos
        conos.
3. Construir el Spline:
    * Se usa Add Spline Point para añadir el punto medio que acaba de calcular como un nuevo punto en el Spline Csv 2.

![function csv 2]({{ site.baseurl }}/assets/images/function_csv2.jpeg)

En el paso final. Se asegura de que el circuito sea un bucle cerrado.

1. Obtener el Punto de Inicio:
    * Tras una comprobación de seguridad (Is Valid), usa Get Location at Spline Point para obtener la posición del primer punto
        (índice 0) del Spline Csv 2 que se acaba de crear.
2. Añadirlo al Final:
    * Llama a Add Spline Point una última vez para añadir un nuevo punto al final del spline. La posición de este nuevo punto
        es la misma que la del punto de inicio.

![function csv 3]({{ site.baseurl }}/assets/images/function_csv3.jpeg)


# Importar el vehiculo a Carla

Finalmente el ultimo paso para preparar los escenarios de nuestra simulacion es tener el vehiculo que vamos a usar dentro del simulador. Para ello yo he seguido esta [guia](https://urjc-deepracer.github.io/docs/importdeepracercarla/)