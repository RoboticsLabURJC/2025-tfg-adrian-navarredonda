# Cómo Hacer que los Conos de un Spline en CARLA Sean Físicos y Reaccionen a Colisiones

En posts anteriores, explique como crear un blueprint para la colocacion de conos de tal 
manera que se forme un circuito, sin embargo esto tiene una gran diferencia con los conos
que colocamos en la vida real, pues en el simulador, al chocar el coche con los conos, 
estos no se movian. Esto se debe a que estaban creados como **Static Mesh Components**
dentro del blueprint de colocacion de conos. 

En este post explico **los cambios que hice y por qué**, para que cualquier actor generado por un spline pueda ser físico y reaccionar a colisiones correctamente.  

## 1. Problema inicial

- Spline generaba conos como **Static Mesh Components**.  
- Estos **no son actores independientes**, por lo que:  
  - No podían simular física.  
  - No reaccionaban a colisiones con el vehículo.  
  - No podían caer o moverse cuando el coche chocaba.  

## 2. Primer cambio: usar actores independientes

Para que los conos reaccionen a colisiones, se necesita que sean **actores independientes**.  

**Solución:**  

- Crear un **Blueprint de cono físico** con:  
  - Un **Static Mesh Component** del cono.  
  - **Mobility = Movable**.  
  - **Simulate Physics = true**.  
  - **Enable Gravity = true**.  
  - **Collision Presets = PhysicsActor**.  

- Modificar el blueprint de colocacion de conos para que **spawnee instancias de este blueprint** en lugar de añadir Static Mesh Components, esto se consigue usando el bloque
**Spawn Actor from Class**.  

## 3. Problema de visibilidad y colisiones al spawnear

Tras cambiar a actores físicos, surgieron dos problemas:

1. **Los conos no se veían en el editor**:  
   - Esto es normal, porque **Spawn Actor from Class** solo genera actores en **tiempo de ejecución (Play)**, mientras que los Static Mesh Components eran visibles siempre en el editor.  

2. **Los conos “saltaban por los aires” al spawnear**:  
   - Esto ocurre cuando los actores físicos se generan **incrustados en otros actores**, como el suelo o entre ellos.  
   - Unreal intenta resolver las colisiones inmediatamente y empuja los conos violentamente.  

## 4. Ajuste de la altura de spawn

Para solucionar que los conos se incrusten en el suelo, ajustamos la coordenada Z al spawnear:  

- Simplemente sumamos un **vector pequeño en Z** al vector del spline:  

```text
VectorSpawn = VectorSpline + (0, 0, 2)
```

- Esto coloca los conos ligeramente por encima del suelo, evitando colisiones instantáneas al aparecer.

## 5. Configuración de Spawn Collision Handling

- Si usábamos *Try to Adjust, Don’t Spawn if Still Colliding*, los conos no aparecían porque estaban demasiado cerca entre sí o del suelo.

- Con *Default / Always Spawn*, los conos aparecían pero saltaban por los aires.

Solucion final:

- Mantener *Collision Handling = Try to Adjust, Don’t Spawn if Still Colliding.*

- Ajustar el Z de spawn ligeramente para que no haya colisión inicial.

## Resultado final

- Todos los conos generados por el spline son *actores independientes*.

- Simulan física correctamente: se mueven, caen y reaccionan a colisiones con el coche.

- La generación es automática a lo largo del spline.
