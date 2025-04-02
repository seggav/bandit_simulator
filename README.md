<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
</head>
  
  
clave unica ______202627____________  

# Problema de Multi-Bandas (Multi-Armed Bandit): Teoría e Implementación

La tarea se entrega por discord antes del miercoles de la siguiente clase. Incluye llenar cuidadosamente en latex todos los snippets mencionados aqui, mas el codigo ya sea con link a colab o al repositorio. No olviden poner su clave unica. La idea es que investiguen, entiendan y proponga una solucion al problema. Utilicen chatgpt y los tutoriales de la tarea (cursor especialmente) para hacer codigo y entender el problema.  

**Nota**  
No pueden utilizar machine learning salvo regresion lineal si asi lo desean (no arboles, deep learning, etc..). 

La proxima clase vamos a continuar con un ejercicio parecido, pero usando cadenas de markov. Vamos a modificar el bandit para que sea mas interesante ante cadenas de markov.  

**Examen**  
El lunes hay examen sobre estos ejercicios a papel y lapiz, la calificacion sera el $min\{examen, ejercicios\}$, si $|examen - ejercicios|<1$ entonces sera el $maximo$. 


## 1. Introducción a los Problemas de Multi-Bandas

### 1.1 Definición y Enunciado del Problema

El problema de Multi-Bandas (MAB, por sus siglas en inglés) es un problema clásico en teoría de la decisión y aprendizaje por refuerzo. Su nombre surge del escenario de un jugador que enfrenta múltiples máquinas tragamonedas (a veces llamadas "bandidos de un solo brazo"), cada una con diferentes probabilidades de recompensa desconocidas. El jugador debe decidir qué máquinas jugar, en qué orden y cuántas veces, para maximizar su recompensa total.

En este modelo:
- Existen $K$ brazos (o acciones) diferentes.
- Cada brazo, cuando se jala, otorga una recompensa extraída de una distribución de probabilidad específica de ese brazo.
- Las distribuciones de recompensa son inicialmente desconocidas para el tomador de decisiones.
- El objetivo es maximizar la recompensa acumulada a lo largo de una serie de jugadas.

El problema captura la disyuntiva fundamental entre **exploración** (probar diferentes brazos para reunir información sobre sus distribuciones de recompensa) y **explotación** (elegir el brazo que actualmente parece ser el mejor).

### 1.2 Dilema de Exploración vs. Explotación

Este dilema está en el corazón del problema de multi-bandas:

- **Exploración**: Seleccionar brazos para aprender más sobre sus distribuciones de recompensa, potencialmente sacrificando recompensas inmediatas.
- **Explotación**: Seleccionar el brazo que actualmente parece ofrecer la mayor recompensa esperada en función de la información reunida hasta el momento.

Equilibrar estos dos aspectos es crucial. Demasiada exploración desperdicia recursos en brazos subóptimos. Demasiada explotación puede impedir descubrir un brazo mejor.

### 1.3 Formulación Matemática General

Formalicemos el problema estándar de bandas estocásticas:

- Sea $K$ el número de brazos.
- Para cada brazo $i \in \{1, 2, \ldots, K\}$, existe una distribución de probabilidad desconocida $\mathcal{D}_i$ con media $\mu_i$.
- En cada paso de tiempo $t \in \{1, 2, \ldots, T\}$:
  - El agente selecciona un brazo $a_t \in \{1, 2, \ldots, K\}$.
  - El agente recibe una recompensa $r_t \sim \mathcal{D}_{a_t}$.
- El objetivo es maximizar la recompensa acumulada $\sum_{t=1}^{T} r_t$.

Alternativamente, el problema puede enmarcarse en términos de minimizar **el arrepentimiento**. El arrepentimiento se define como la diferencia entre la recompensa obtenida al seleccionar siempre el brazo óptimo y la recompensa realmente obtenida por el agente:

$\text{Regret}(T) = T \cdot \max_{i} \mu_i - \mathbb{E}\left[\sum_{t=1}^{T} r_t\right]$

## 2. Escenarios de Información en Nuestro Entorno de Bandas

En nuestro entorno de multi-bandas, exploramos tres escenarios de información distintos, cada uno proporcionando al agente diferentes niveles de conocimiento:

### 2.1 Escenario de Información Completa

En este escenario, el agente observa:
- El número de turno actual.
- El número total de turnos T.
- La probabilidad de recompensa para el brazo 1 (p1).
- El historial completo de acciones y recompensas pasadas.

Este es el escenario más informativo, ya que el agente conoce la probabilidad de uno de los brazos directamente y puede inferir la del otro con base en las recompensas observadas.

### 2.2 Escenario de Información Parcial

En este escenario, el agente observa:
- El número de turno actual.
- El número total de turnos T.
- La probabilidad de recompensa para el brazo 1 (p1).
- El historial de acciones y recompensas pasadas.

El agente conoce la probabilidad de un brazo pero debe aprender la del otro a través de la experimentación.

### 2.3 Escenario de Solo Recompensa

En este escenario, el agente observa:
- El número de turno actual.
- El historial de acciones y recompensas pasadas.

Este es el escenario más desafiante porque:
1. El agente no conoce la probabilidad de ninguno de los dos brazos.
2. El agente no conoce el número total de turnos T.

El agente debe aprender las probabilidades de ambos brazos mediante la experimentación y no puede optimizar su estrategia en función de la duración conocida del juego.

## 3. Entornos de Bandas en Nuestro Playground

Nuestro entorno implementa cuatro tipos diferentes de entornos de multi-bandas, cada uno con características distintas que afectan cómo cambian las probabilidades de los brazos a lo largo del tiempo.

### 3.1 Entorno de Banda Fija

#### Descripción
En el entorno de Banda Fija, cada brazo tiene una probabilidad constante de recompensa durante todo el juego. Estas probabilidades se asignan aleatoriamente al inicio de cada juego (uniforme entre 0.01 y 0.99) y permanecen sin cambios.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- Probabilidades fijas: $p_1, p_2 \in [0.01, 0.99]$
- En el turno $t$, al seleccionar el brazo $a$:
  - Se recibe recompensa $r_t = 1$ con probabilidad $p_{a+1}$
  - Se recibe recompensa $r_t = 0$ con probabilidad $1 - p_{a+1}$

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Fija con horizonte de tiempo conocido T = 100. ¿Cuál es la función objetivo? ¿Cuáles son las restricciones? ¿Cuál es la política óptima?
```latex
El problema vendría siendo el equilibrar la exploración y la explotación, para así 

%%%% Función Objetivo:
% Maximizar la recompensa acumulada total en \(T\) tiros:
\[
\max_{\pi} \, \mathbb{E}\left[ \sum_{t=1}^{T} r_t \right] \quad \text{con } T = 100,
\]
donde \(\pi\) es la política de selección de brazos a lo largo de los tiros.

%%%% Restricciones:
\begin{itemize}
  \item \textbf{Selección única:} En cada turno \(t\), se debe elegir un único brazo \(a \in \{0, 1\}\).
  \item \textbf{Horizonte fijo:} El número total de tiros es \(T = 100\).
  \item \textbf{Actualización del estado:} Tras cada tiro, se actualiza el estado \(s\) (por ejemplo, el conteo de éxitos y fracasos para cada brazo) que influye en la decisión en turnos futuros.
\end{itemize}

%%%% Política Óptima:
% La política óptima selecciona, en cada tiro \(t\), el brazo \(a\) que maximiza la suma de la recompensa inmediata esperada y la recompensa acumulada esperada para los tiros restantes.
% Esto se expresa mediante la ecuación de Bellman:
\[
V(s,t) = \max_{a \in \{0, 1\}} \left\{ \mathbb{E}\left[ r(a) \mid s \right] + V(s', t-1) \right\}, \quad \text{con } V(s,0) = 0,
\]
donde:
\begin{itemize}
  \item \(\mathbb{E}\left[ r(a) \mid s \right]\) es la recompensa inmediata esperada al elegir el brazo \(a\) dado el estado actual \(s\).
  \item \(V(s', t-1)\) es la función de valor que representa la recompensa acumulada esperada en los \(t-1\) tiros restantes, partiendo del nuevo estado \(s'\) tras tomar la acción \(a\).
\end{itemize}
La política óptima en cada tiro se determina como:
\[
a^* = \arg\max_{a \in \{0, 1\}} \left\{ \mathbb{E}\left[ r(a) \mid s \right] + V(s', t-1) \right\}.
\]
Esta formulación permite equilibrar la \textbf{explotación} (elegir el brazo con mejor rendimiento estimado) y la \textbf{exploración} (probar el otro brazo para aprender sobre su probabilidad de éxito).





```

#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Fija con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cómo afecta el horizonte de tiempo aleatorio la estrategia óptima?
```latex

%%%% Formulación Matemática
% Dos brazos: \(a \in \{0, 1\}\)
% Probabilidades fijas: \(p_1, p_2 \in [0.01, 0.99]\)
% Horizonte de tiempo aleatorio: \(T \sim \text{Uniform}(1,300)\)

%%%% Definición de recompensa:
% En el turno \(t\), al seleccionar el brazo \(a\):
%   - Se recibe recompensa \(r_t = 1\) con probabilidad \(p_{a+1}\)
%   - Se recibe recompensa \(r_t = 0\) con probabilidad \(1 - p_{a+1}\)

%%%% Función Objetivo:
% Maximizar la recompensa acumulada total en un horizonte de tiempo aleatorio:
\[
\max_{\pi} \, \mathbb{E}\left[ \sum_{t=1}^{T} r_t \right], \quad \text{con } T \sim \text{Uniform}(1,300),
\]
donde \(\pi\) es la política de selección de brazos.

%%%% Restricciones:
\begin{itemize}
  \item \textbf{Selección única por turno:} En cada turno \(t\), se debe elegir un único brazo \(a \in \{0, 1\}\).
  \item \textbf{Horizonte aleatorio:} El número total de tiros \(T\) es una variable aleatoria con distribución uniforme en el intervalo \([1,300]\).
  \item \textbf{Actualización del estado:} Tras cada tiro se actualiza el estado \(s\) (por ejemplo, el conteo de éxitos y fracasos para cada brazo) que influirá en la toma de decisiones futuras.
\end{itemize}

%%%% Política Óptima:
% Debido a que el horizonte de tiempo es aleatorio, la política óptima debe maximizar la recompensa acumulada esperada considerando la incertidumbre sobre el número total de tiros.
% La ecuación de Bellman se ajusta para incorporar esta incertidumbre:
\[
V(s,t) = \max_{a \in \{0, 1\}} \left\{ \mathbb{E}\left[ r(a) \mid s \right] + \mathbb{E}\left[ V(s',t-1) \mid T \ge t \right] \right\}, \quad \text{con } V(s,0) = 0.
\]
Aquí:
\begin{itemize}
  \item \(\mathbb{E}\left[ r(a) \mid s \right]\) es la recompensa inmediata esperada al seleccionar el brazo \(a\) dado el estado actual \(s\).
  \item \(\mathbb{E}\left[ V(s',t-1) \mid T \ge t \right]\) es el valor esperado de la recompensa futura, condicionado a que aún queden tiros (es decir, \(T \ge t\)).
\end{itemize}
La política óptima en cada turno se define como:
\[
a^* = \arg\max_{a \in \{0, 1\}} \left\{ \mathbb{E}\left[ r(a) \mid s \right] + \mathbb{E}\left[ V(s',t-1) \mid T \ge t \right] \right\}.
\]

%%%% Impacto del Horizonte de Tiempo Aleatorio:
% El hecho de que \(T\) sea aleatorio afecta la estrategia óptima de la siguiente manera:
\begin{itemize}
  \item \textbf{Incertidumbre sobre la duración:} Al no conocer el número exacto de tiros, la estrategia debe considerar la probabilidad de que el experimento finalice en cada turno.
  \item \textbf{Equilibrio exploración-explotación:} 
    \begin{itemize}
      \item Si existe una probabilidad significativa de que \(T\) sea pequeño (horizonte corto), se favorece la explotación, ya que se dispone de menos oportunidades para aprender.
      \item Si es probable que \(T\) sea mayor, se puede permitirse una mayor exploración para mejorar las estimaciones de \(p_2\) y obtener mejores decisiones en el futuro.
    \end{itemize}
  \item \textbf{Planificación en valor esperado:} La optimización se realiza considerando el valor esperado del número de tiros futuros, integrando la distribución uniforme de \(T\). Esto implica que la política se ajusta dinámicamente en función de la probabilidad de que queden tiros adicionales.
\end{itemize}

%%%% Resumen:
% El problema de decisión para la Banda Fija con un horizonte de tiempo aleatorio \(T \sim \text{Uniform}(1,300)\) se define mediante:
%
% \[
% \max_{\pi} \, \mathbb{E}\left[ \sum_{t=1}^{T} r_t \right],
% \]
% sujeto a:
% \begin{itemize}
%   \item La selección de un único brazo \(a \in \{0, 1\}\) por turno.
%   \item La actualización del estado \(s\) tras cada tiro.
% \end{itemize}
%
% La política óptima se obtiene resolviendo la ecuación de Bellman modificada, que integra la incertidumbre sobre el número total de tiros. Este horizonte aleatorio influye en la estrategia al:
% \begin{itemize}
%   \item Incentivar la explotación temprana en escenarios de posible horizonte corto.
%   \item Permitir una mayor exploración cuando la probabilidad de un horizonte largo es considerable.
%   \item Requerir la planificación basada en el valor esperado de tiros futuros.
% \end{itemize}
```

### 3.2 Entorno de Banda Periódica

#### Descripción
En el entorno de Banda Periódica, la probabilidad de recompensa de cada brazo cambia cada k turnos (por defecto, k=10). En cada punto de cambio, se asignan nuevas probabilidades aleatorias (uniforme entre 0.01 y 0.99) a ambos brazos.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- En el turno $t$, las probabilidades son:
  - $p_1(t) = p_1^{\lfloor t/k \rfloor}$, donde $p_1^j \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) = p_2^{\lfloor t/k \rfloor}$, donde $p_2^j \sim \text{Uniform}(0.01, 0.99)$
- El superíndice $j = \lfloor t/k \rfloor$ indica el número de "período".
- En cada punto de cambio (cuando $t$ es divisible por $k$), se asignan nuevos valores aleatorios.

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Periódica con horizonte de tiempo conocido T = 100 y período k = 10. ¿Cómo abordarías la búsqueda de una estrategia óptima? ¿Qué información adicional sería valiosa rastrear?
```latex
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\begin{document}

\section*{Problema de Decisión para la Banda Periódica}

\subsection*{Formulación Matemática}
\begin{itemize}
  \item \textbf{Brazos y Recompensas:}  
  Se tienen dos brazos, identificados por \(a \in \{0,1\}\). Cada brazo tiene una probabilidad fija de éxito:
  \begin{itemize}
    \item Para el brazo \(a=0\): la probabilidad de éxito es \(p_1\).
    \item Para el brazo \(a=1\): la probabilidad de éxito es \(p_2\).
  \end{itemize}
  En el turno \(t\), al seleccionar el brazo \(a\):
  \[
  r_t = 
  \begin{cases}
    1, & \text{con probabilidad } p_{a+1},\\[1ex]
    0, & \text{con probabilidad } 1-p_{a+1}.
  \end{cases}
  \]
  
  \item \textbf{Horizonte de Tiempo y Periodicidad:}  
  El experimento se lleva a cabo en \(T = 100\) turnos, y las probabilidades de éxito presentan una estructura periódica con período \(k = 10\). Es decir, para cada brazo, la probabilidad de éxito se repite cada 10 turnos:
  \[
  p_{a+1}(t) = p_{a+1}(t+10), \quad \forall t.
  \]
\end{itemize}

\subsection*{Función Objetivo}
El objetivo es maximizar la recompensa acumulada total en los 100 turnos:
\[
\max_{\pi} \; \mathbb{E}\left[ \sum_{t=1}^{100} r_t \right],
\]
donde \(\pi\) representa la política o estrategia de selección de brazos.

\subsection*{Restricciones}
\begin{enumerate}[label=\alph*.]
  \item \textbf{Selección Única:}  
  En cada turno \(t\), se debe seleccionar un único brazo \(a \in \{0,1\}\).
  
  \item \textbf{Actualización del Estado:}  
  El estado \(s\) incluye la información acumulada, como los conteos de éxitos y fracasos para cada brazo, y debe incorporar la fase actual del ciclo, definida como:
  \[
  \phi = t \mod 10.
  \]
\end{enumerate}

\subsection*{Búsqueda de una Estrategia Óptima}
Para abordar la búsqueda de una estrategia óptima se sugiere:
\begin{enumerate}
  \item \textbf{Incorporar la Periodicidad en el Estado:}  
  Definir el estado como \( s = (\text{conteos por brazo}, \phi) \), donde \(\phi = t \mod 10\). Esto permite distinguir entre las distintas fases del ciclo y aprovechar la repetitividad de las probabilidades.
  
  \item \textbf{Aplicar Programación Dinámica o Aprendizaje por Refuerzo:}  
  Utilizar la ecuación de Bellman adaptada a la periodicidad:
  \[
  V(s,\phi,t) = \max_{a \in \{0,1\}} \left\{ \mathbb{E}\left[r(a) \mid s,\phi\right] + V\Big(s',(\phi+1) \mod 10, t-1\Big) \right\},
  \]
  con condición terminal:
  \[
  V(s,\phi,0) = 0.
  \]
  
  \item \textbf{Balance entre Exploración y Explotación:}  
  Aprovechar la información acumulada por fase para decidir:
  \begin{itemize}
    \item \emph{Exploración:} Probar en fases donde se tenga poca información.
    \item \emph{Explotación:} Seleccionar el brazo que actualmente muestra mejor desempeño en aquellas fases donde se cuenta con datos confiables.
  \end{itemize}
\end{enumerate}

\subsection*{Información Adicional Valiosa a Rastrear}
Para mejorar la estrategia, es valioso monitorizar:
\begin{itemize}
  \item \textbf{Fase del Ciclo (\(\phi = t \mod 10\)):}  
  Permite agrupar la experiencia obtenida en turnos equivalentes dentro de cada ciclo.
  
  \item \textbf{Estadísticas por Fase y por Brazo:}  
  Registrar el número de selecciones, éxitos y fracasos para cada brazo en cada fase del ciclo.
  
  \item \textbf{Estimaciones y Uncertidumbre:}  
  Actualizar las estimaciones de \(p_1\) y \(p_2\) para cada fase, junto con medidas de incertidumbre (por ejemplo, varianzas) que permitan ajustar el balance entre exploración y explotación.
  
  \item \

```
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Periódica con horizonte de tiempo desconocido T ~ Uniform(1, 300) y período k = 10. ¿Cómo interactúa la aleatoriedad en T con la naturaleza periódica del entorno?
```latex
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\begin{document}

\section*{Problema de Decisión: Banda Periódica con Horizonte Aleatorio}

Consideremos el siguiente entorno:
\begin{itemize}
    \item \textbf{Brazos y Recompensas:}  
    Tenemos dos brazos, \(a \in \{0,1\}\). Al seleccionar el brazo \(a\) en el turno \(t\), se obtiene una recompensa:
    \[
    r_t = \begin{cases}
    1, & \text{con probabilidad } p_{a+1}(t),\\[1ex]
    0, & \text{con probabilidad } 1 - p_{a+1}(t),
    \end{cases}
    \]
    donde las probabilidades son periódicas con período \(k=10\):
    \[
    p_{a+1}(t) = p_{a+1}(t+10).
    \]
    
    \item \textbf{Horizonte de Tiempo Aleatorio:}  
    El número total de turnos \(T\) es desconocido y se distribuye uniformemente:
    \[
    T \sim \text{Uniform}(1,300).
    \]
\end{itemize}

\subsection*{Función Objetivo}
El objetivo es maximizar la recompensa acumulada hasta que finalice el proceso:
\[
\max_{\pi} \; \mathbb{E}\left[\sum_{t=1}^{T} r_t\right],
\]
donde \(\pi\) representa la política (la secuencia de decisiones) que elige en cada turno un brazo \(a \in \{0,1\}\).

\subsection*{Estado y Consideración de la Periodicidad}
Dado que las probabilidades de éxito se repiten cada \(10\) turnos, es crucial incorporar la fase del ciclo en el estado. Definimos:
\[
\phi = t \mod 10,
\]
de manera que el estado \(s\) se actualice incluyendo tanto la información acumulada (por ejemplo, conteos de éxitos y fracasos por brazo) como la fase \(\phi\).

\subsection*{Interacción entre la Aleatoriedad en \(T\) y la Naturaleza Periódica del Entorno}
La incertidumbre en la duración del proceso interactúa con la periodicidad de la siguiente manera:
\begin{itemize}
    \item \textbf{Incertidumbre del Horizonte:}  
    Al no conocer de antemano el número total de turnos, la política debe estar preparada para escenarios en los que \(T\) sea corto o largo. En un horizonte corto, se tiende a priorizar la recompensa inmediata (explotación), mientras que en un horizonte largo es posible invertir en explorar para aprender la estructura periódica y mejorar las decisiones en turnos futuros.
    
    \item \textbf{Dependencia de la Fase:}  
    Dado que las probabilidades de recompensa varían cíclicamente, la decisión óptima puede depender fuertemente de la fase del ciclo \(\phi\). Por ejemplo, un brazo que rinde mejor en la fase \(\phi=3\) debe seleccionarse preferentemente cuando se esté en esa fase.
    
    \item \textbf{Planificación Dinámica:}  
    La política óptima se puede abordar mediante una versión modificada de la ecuación de Bellman que considere tanto la incertidumbre en \(T\) como la periodicidad:
    \[
    V(s,\phi,t) = \max_{a \in \{0,1\}} \left\{ \mathbb{E}\left[r(a) \mid s,\phi\right] + \mathbb{E}\left[V(s',(\phi+1) \mod 10, t-1)\right] \right\},
    \]
    con la condición terminal \(V(s,\phi,0)=0\), donde \(t\) es el número de turnos restantes. Aquí, la expectativa sobre \(V(s',(\phi+1) \mod 10, t-1)\) incorpora la probabilidad de que el proceso continúe en función del horizonte aleatorio.
\end{itemize}

\subsection*{Resumen}
El problema de decisión para la Banda Periódica con horizonte aleatorio \(T \sim \text{Uniform}(1,300)\) y período \(k=10\) se define considerando:
\begin{itemize}
    \item La estructura de recompensas que varía cíclicamente, con \(p_{a+1}(t) = p_{a+1}(t+10)\).
    \item Un horizonte de tiempo incierto, que exige un balance dinámico entre exploración y explotación.
    \item La necesidad de incorporar la fase del ciclo \(\phi = t \mod 10\) en el estado para adaptar la estrategia a la variabilidad periódica.
\end{itemize}
La aleatoriedad en \(T\) obliga a la política a valorar continuamente la recompensa inmediata frente a la futura, mientras que la periodicidad requiere un seguimiento detallado de las fases para explotar las ventajas temporales específicas de cada brazo.

\end{document}
```
### 3.3 Entorno de Banda Dinámica

#### Descripción
En el entorno de Banda Dinámica, las probabilidades de recompensa para ambos brazos cambian en cada turno. Cada turno se asignan probabilidades aleatorias completamente nuevas (uniforme entre 0.01 y 0.99) a ambos brazos.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- En el turno $t$, las probabilidades son:
  - $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- Se generan nuevos valores aleatorios en cada turno.

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Dinámica con horizonte de tiempo conocido T = 100. ¿Hay una forma significativa de aprender de observaciones pasadas en este entorno? ¿Cuál sería la estrategia óptima?
```latex
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\begin{document}

\section*{Problema de Decisión: Banda Dinámica con Horizonte Fijo \(T = 100\)}

\subsection*{Descripción y Formulación}
En el entorno de Banda Dinámica se tiene lo siguiente:
\begin{itemize}
    \item \textbf{Brazos:} \(a \in \{0,1\}\).
    \item \textbf{Probabilidades de Recompensa:}  
    En cada turno \(t\), las probabilidades de éxito se generan de forma independiente y aleatoria:
    \[
    p_1(t) \sim \text{Uniform}(0.01, 0.99) \quad \text{y} \quad p_2(t) \sim \text{Uniform}(0.01, 0.99).
    \]
    Es decir, en cada turno se asignan nuevos valores a \(p_1(t)\) y \(p_2(t)\).
    \item \textbf{Recompensa:}  
    Al seleccionar el brazo \(a\) en el turno \(t\), se obtiene:
    \[
    r_t = 
    \begin{cases}
    1, & \text{con probabilidad } p_{a+1}(t),\\[1ex]
    0, & \text{con probabilidad } 1 - p_{a+1}(t).
    \end{cases}
    \]
    \item \textbf{Horizonte de Tiempo:}  
    Se realizan \(T = 100\) turnos.
\end{itemize}

La función objetivo es maximizar la recompensa acumulada durante los \(100\) turnos:
\[
\max_{\pi} \; \mathbb{E}\left[\sum_{t=1}^{100} r_t\right],
\]
donde \(\pi\) representa la política de selección de brazos.

\subsection*{Aprendizaje de Observaciones Pasadas}
Dado que en cada turno se generan nuevos valores aleatorios para \(p_1(t)\) y \(p_2(t)\) de forma independiente, no existe correlación entre los valores de turnos anteriores y los del turno actual. Por lo tanto, la información obtenida en turnos previos no tiene valor predictivo para el comportamiento futuro de los brazos. Cada turno es, efectivamente, un nuevo experimento independiente.

\subsection*{Estrategia Óptima}
La estrategia óptima depende de la información disponible:

\begin{enumerate}[label=\alph*.]
    \item \textbf{Sin Observación de Parámetros:}  
    En el escenario típico, el agente no puede observar los valores actuales de \(p_1(t)\) y \(p_2(t)\) antes de tomar la decisión. Dado que el valor esperado de la Uniform(0.01, 0.99) es aproximadamente 0.5 para ambos brazos, no existe ventaja en preferir uno sobre el otro.  
    \textbf{Estrategia:} Seleccionar los brazos de manera equiprobable (por ejemplo, cada uno con probabilidad 0.5).
    
    \item \textbf{Con Observación de Parámetros:}  
    Si, en una situación poco realista, el agente pudiera conocer los valores actuales de \(p_1(t)\) y \(p_2(t)\) antes de elegir, la estrategia óptima sería seleccionar el brazo con la mayor probabilidad de éxito en ese turno.
\end{enumerate}

Dado que la configuración típica del problema supone que los parámetros subyacentes son desconocidos, la conclusión es que en el entorno de Banda Dinámica no se puede aprender de observaciones pasadas para mejorar las decisiones en turnos futuros, puesto que cada turno es independiente.

\end{document}

```
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Dinámica con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cambia significativamente el enfoque óptimo en este entorno altamente dinámico si el horizonte de tiempo es desconocido?
```latex
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\begin{document}

\section*{Problema de Decisión: Banda Dinámica con Horizonte Desconocido \(T \sim \text{Uniform}(1,300)\)}

\subsection*{Formulación del Problema}
En el entorno de Banda Dinámica se tienen dos brazos \(a \in \{0,1\}\). En cada turno \(t\), las probabilidades de éxito se generan de forma independiente y aleatoria:
\[
p_1(t) \sim \text{Uniform}(0.01, 0.99) \quad \text{y} \quad p_2(t) \sim \text{Uniform}(0.01, 0.99).
\]
La recompensa al seleccionar el brazo \(a\) en el turno \(t\) es:
\[
r_t = 
\begin{cases}
1, & \text{con probabilidad } p_{a+1}(t),\\[1ex]
0, & \text{con probabilidad } 1-p_{a+1}(t).
\end{cases}
\]
El horizonte de tiempo \(T\) es desconocido y se distribuye uniformemente:
\[
T \sim \text{Uniform}(1,300).
\]
El objetivo es maximizar la recompensa acumulada:
\[
\max_{\pi} \; \mathbb{E}\left[\sum_{t=1}^{T} r_t\right],
\]
donde \(\pi\) es la política de selección de brazos.

\subsection*{Discusión: Impacto del Horizonte Desconocido en un Entorno Altamente Dinámico}
\begin{itemize}
    \item \textbf{Independencia de Turno:}  
    En este entorno, en cada turno se generan nuevos valores aleatorios para \(p_1(t)\) y \(p_2(t)\). Esto implica que las observaciones de turnos anteriores no tienen valor predictivo para el turno actual, ya que cada turno es un nuevo experimento independiente.
    
    \item \textbf{Decisión Óptima en Cada Turno:}  
    La decisión en cada turno depende únicamente de la información disponible en ese momento. Si el agente tuviera acceso a los valores actuales de \(p_1(t)\) y \(p_2(t)\) (lo cual es poco común en problemas de bandido), la estrategia óptima sería elegir el brazo con la mayor probabilidad de éxito en ese turno. Si no, en ausencia de esa información, la elección se reduce a una decisión sin sesgo (por ejemplo, seleccionar cada brazo con probabilidad 0.5).
    
    \item \textbf{Horizonte Desconocido:}  
    El hecho de que \(T\) sea desconocido y se distribuya uniformemente en \([1,300]\) afecta el número total de turnos disponibles, pero no altera la estructura del problema en cada turno. Debido a la independencia entre turnos, la incertidumbre sobre el horizonte no introduce nuevos beneficios en términos de aprendizaje o planificación a futuro.
\end{itemize}

\subsection*{Conclusión}
En resumen, en el entorno de Banda Dinámica con un horizonte de tiempo desconocido \(T \sim \text{Uniform}(1,300)\):
\begin{enumerate}
    \item Cada turno es independiente, por lo que no se puede aprender de las observaciones pasadas para influir en las decisiones futuras.
    \item La decisión óptima en cada turno se basa exclusivamente en la información del turno actual.
    \item La incertidumbre en \(T\) afecta únicamente el número total de turnos, pero dado que cada turno es independiente, no cambia significativamente el enfoque óptimo en comparación con el caso de un horizonte fijo.
\end{enumerate}
Por lo tanto, en un entorno altamente dinámico como este, la estrategia óptima sigue siendo:
\begin{itemize}
    \item Si se conocen los valores actuales de \(p_1(t)\) y \(p_2(t)\): elegir el brazo con la mayor probabilidad de éxito.
    \item Si no se conocen: seleccionar los brazos de forma equiprobable, ya que ambos tienen el mismo valor esperado.
\end{itemize}

\end{document}

```
### 3.4 Entorno de Banda Totalmente Aleatorio

#### Descripción
En el entorno de Banda Totalmente Aleatorio, las probabilidades de los brazos se inicializan de forma aleatoria y luego cambian aleatoriamente con una pequeña probabilidad (5%) en cada turno. Esto crea un entorno donde los cambios son impredecibles pero ocurren con menos frecuencia que en el entorno Dinámico.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- Probabilidades iniciales: $p_1(0), p_2(0) \sim \text{Uniform}(0.01, 0.99)$
- En el turno $t > 0$, con probabilidad 0.05:
  - $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- De lo contrario (con probabilidad 0.95):
  - $p_1(t) = p_1(t-1)$
  - $p_2(t) = p_2(t-1)$

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Totalmente Aleatoria con horizonte de tiempo conocido T = 100. ¿Cómo equilibrarías la exploración y explotación sabiendo que las probabilidades de los brazos podrían cambiar repentinamente?
```latex
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\begin{document}

\section*{Problema de Decisión: Banda Totalmente Aleatoria con Horizonte \(T = 100\)}

\subsection*{Descripción y Formulación}
En este entorno, se consideran dos brazos \(a \in \{0,1\}\) con las siguientes características:
\begin{itemize}
    \item \textbf{Probabilidades Iniciales:}  
    \[
    p_1(0),\, p_2(0) \sim \text{Uniform}(0.01, 0.99).
    \]
    
    \item \textbf{Evolución de las Probabilidades:}  
    Para cada turno \(t > 0\):
    \begin{itemize}
        \item Con probabilidad \(0.05\), se actualizan las probabilidades de forma independiente:
        \[
        p_1(t) \sim \text{Uniform}(0.01, 0.99), \quad p_2(t) \sim \text{Uniform}(0.01, 0.99).
        \]
        \item Con probabilidad \(0.95\), las probabilidades permanecen iguales a las del turno anterior:
        \[
        p_1(t) = p_1(t-1), \quad p_2(t) = p_2(t-1).
        \]
    \end{itemize}
    
    \item \textbf{Recompensa:}  
    Al seleccionar el brazo \(a\) en el turno \(t\), se obtiene:
    \[
    r_t = 
    \begin{cases}
    1, & \text{con probabilidad } p_{a+1}(t),\\[1ex]
    0, & \text{con probabilidad } 1 - p_{a+1}(t).
    \end{cases}
    \]
    
    \item \textbf{Horizonte de Tiempo:}  
    Se realizan \(T = 100\) turnos.
\end{itemize}

La función objetivo es maximizar la recompensa acumulada:
\[
\max_{\pi} \; \mathbb{E}\left[\sum_{t=1}^{100} r_t\right],
\]
donde \(\pi\) es la política de selección de brazos.

\subsection*{Equilibrio entre Exploración y Explotación}
En este entorno, las probabilidades de recompensa se mantienen constantes en la mayoría de los turnos (95\%), pero pueden cambiar abruptamente con probabilidad 0.05 en cada turno. Esto implica que:
\begin{itemize}
    \item \textbf{Explotación:}  
    Si un brazo presenta un buen desempeño durante un período prolongado, se tenderá a explotarlo para maximizar la recompensa inmediata.
    
    \item \textbf{Exploración:}  
    Sin embargo, debido a la posibilidad de cambios repentinos en las probabilidades, es crucial mantener un cierto nivel de exploración constante para detectar rápidamente cualquier cambio en el comportamiento de los brazos. Esto permite:
    \begin{itemize}
        \item Actualizar las estimaciones de \(p_1(t)\) y \(p_2(t)\) de forma oportuna.
        \item Evitar quedarse atrapado en una estrategia que resultó ser óptima en el pasado, pero que ya no lo es tras un cambio.
    \end{itemize}
\end{itemize}

\subsection*{Estrategia Óptima}
Dada la naturaleza de este entorno, una estrategia óptima debe incluir:
\begin{enumerate}[label=\alph*.]
    \item \textbf{Exploración Persistente:}  
    Utilizar un método que garantice una exploración no nula en cada turno (por ejemplo, un \(\varepsilon\)-greedy con \(\varepsilon > 0\) constante o estrategias basadas en ventanas deslizantes que den mayor peso a las observaciones recientes).
    
    \item \textbf{Adaptación Rápida:}  
    Incorporar mecanismos de detección de cambios, que permitan actualizar rápidamente las estimaciones de las probabilidades cuando se observe un cambio abrupto.
    
    \item \textbf{Balance Dinámico:}  
    Ajustar dinámicamente el balance entre exploración y explotación, reconociendo que:
    \begin{itemize}
        \item Durante períodos de estabilidad (cuando las probabilidades no cambian), se puede favorecer la explotación.
        \item Cuando se sospecha un cambio, se debe aumentar la exploración para revaluar la situación.
    \end{itemize}
\end{enumerate}

\subsection*{Conclusión}
En el entorno de Banda Totalmente Aleatoria con \(T = 100\), el enfoque óptimo no cambia de manera significativa al conocer el horizonte, ya que cada turno es afectado por la posibilidad de un cambio repentino en las probabilidades. Por ello, la clave está en mantener una exploración continua y adaptativa que permita detectar y responder a estos cambios, sin dejar de aprovechar las oportunidades cuando se identifique un brazo superior.

\end{document}

```
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Totalmente Aleatoria con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cómo interactúan las dos formas de aleatoriedad (en las probabilidades de los brazos y en el horizonte de tiempo)?
```latex
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{enumitem}
\begin{document}

\section*{Problema de Decisión: Banda Totalmente Aleatoria con Horizonte Desconocido \(T \sim \text{Uniform}(1,300)\)}

\subsection*{Formulación del Problema}
Consideremos un entorno de bandido con dos brazos \(a \in \{0,1\}\) que se caracteriza por lo siguiente:
\begin{itemize}
    \item \textbf{Probabilidades Iniciales:}  
    \[
    p_1(0),\, p_2(0) \sim \text{Uniform}(0.01,0.99).
    \]
    
    \item \textbf{Evolución de las Probabilidades:}  
    Para cada turno \(t > 0\):
    \begin{itemize}
        \item Con probabilidad \(0.05\), se actualizan las probabilidades de forma independiente:
        \[
        p_1(t) \sim \text{Uniform}(0.01,0.99), \quad p_2(t) \sim \text{Uniform}(0.01,0.99).
        \]
        \item Con probabilidad \(0.95\), las probabilidades se mantienen iguales a las del turno anterior:
        \[
        p_1(t) = p_1(t-1), \quad p_2(t) = p_2(t-1).
        \]
    \end{itemize}
    
    \item \textbf{Horizonte de Tiempo Aleatorio:}  
    El número total de turnos \(T\) es incierto y se distribuye uniformemente:
    \[
    T \sim \text{Uniform}(1,300).
    \]
    
    \item \textbf{Recompensa:}  
    En cada turno \(t\), al seleccionar el brazo \(a\), se obtiene la recompensa:
    \[
    r_t = 
    \begin{cases}
    1, & \text{con probabilidad } p_{a+1}(t),\\[1ex]
    0, & \text{con probabilidad } 1-p_{a+1}(t).
    \end{cases}
    \]
\end{itemize}

La función objetivo es maximizar la recompensa acumulada:
\[
\max_{\pi} \; \mathbb{E}\left[ \sum_{t=1}^{T} r_t \right],
\]
donde \(\pi\) es la política de selección de brazos.

\subsection*{Interacción de las Dos Fuentes de Aleatoriedad}
En este entorno se presentan dos formas de aleatoriedad:

\begin{enumerate}[label=\alph*.]
    \item \textbf{Aleatoriedad en las Probabilidades de los Brazos:}  
    Las probabilidades \(p_1(t)\) y \(p_2(t)\) pueden cambiar abruptamente en cada turno con probabilidad 0.05. Esto significa que la información acumulada en turnos previos puede volverse obsoleta si ocurre un cambio.
    
    \item \textbf{Aleatoriedad en el Horizonte de Tiempo:}  
    El número total de turnos \(T\) es incierto, variando entre 1 y 300. Esta incertidumbre afecta la cantidad de oportunidades disponibles para acumular recompensas.
\end{enumerate}

Estas dos fuentes de incertidumbre interactúan de la siguiente manera:
\begin{itemize}
    \item Si el horizonte \(T\) resulta corto, el impacto de un cambio en las probabilidades es crítico, ya que hay pocas oportunidades para recuperarse de una mala elección. En este caso, la política debe priorizar la obtención de recompensas inmediatas.
    
    \item Si el horizonte \(T\) es largo, se dispondrá de más turnos para adaptarse a cambios repentinos en las probabilidades. Sin embargo, dado que los cambios ocurren con una probabilidad del 5\% en cada turno, la información histórica puede volverse rápidamente irrelevante, exigiendo una exploración continua.
    
    \item En ambos escenarios, la estrategia debe balancear la explotación de un brazo que parece óptimo en el momento con la exploración para detectar posibles cambios en las probabilidades.
\end{itemize}

\subsection*{Conclusión}
En el entorno de Banda Totalmente Aleatoria con \(T \sim \text{Uniform}(1,300)\), la interacción de la aleatoriedad en las probabilidades de los brazos y en el horizonte de tiempo implica que:
\begin{itemize}
    \item La política debe mantener una exploración continua para detectar rápidamente cambios repentinos en las probabilidades.
    \item La incertidumbre en \(T\) obliga a no depender excesivamente de información histórica, ya que el número de turnos disponibles para aprovechar el aprendizaje es variable.
    \item El equilibrio entre exploración y explotación se ajusta dinámicamente: en un horizonte potencialmente corto se prioriza la recompensa inmediata, mientras que en un horizonte largo se puede permitir una mayor exploración, siempre con la cautela de que la estabilidad de las probabilidades es limitada.
\end{itemize}

Esta formulación resalta la necesidad de estrategias robustas que se adapten tanto a la volatilidad en las probabilidades como a la incertidumbre en el número total de turnos disponibles.

\end{document}

```
## 4. Implementación de Agentes

En nuestro entorno, implementarás tres tipos de agentes correspondientes a los tres escenarios de información descritos anteriormente. Esto es lo que cada agente debe manejar:

### 4.1 Agente de Información Completa

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'total_turns': int,         # Número total de turnos en el juego
    'p1': float,                # Probabilidad de recompensa del brazo 1
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...], # Recompensas pasadas
        'p1': [float, ...],      # Historial de probabilidades del brazo 1
        'p2': [float, ...]       # Historial de probabilidades del brazo 2 (solo para evaluación)
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

### 4.2 Agente de Información Parcial

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'total_turns': int,         # Número total de turnos en el juego
    'p1': float,                # Probabilidad de recompensa del brazo 1
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...]  # Recompensas pasadas
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

### 4.3 Agente de Solo Recompensa

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...]  # Recompensas pasadas
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

## 5. Métricas de Rendimiento

El entorno evalúa el rendimiento de los agentes usando varias métricas clave:

### 5.1 Recompensa Promedio

Esta es la recompensa media obtenida por turno, calculada como:

$\text{Recompensa Promedio} = \frac{1}{T} \sum_{t=1}^{T} r_t$

Esta métrica mide directamente qué tan bien el agente está maximizando su función objetivo. Valores más altos indican un mejor rendimiento.

### 5.2 Porcentaje de Acciones Óptimas

Esta métrica mide el porcentaje de veces que el agente seleccionó el brazo con la mayor probabilidad de recompensa:

$\text{Acciones Óptimas (\%)} = \frac{100}{T} \sum_{t=1}^{T} \mathbf{1}\{a_t = \arg\max_i p_i(t)\}$

Donde $\mathbf{1}$ es la función indicadora que vale 1 cuando la condición es verdadera y 0 en caso contrario.

Esta métrica muestra con qué frecuencia el agente elige el mejor brazo, independientemente de la recompensa real recibida. Valores más altos indican una mejor selección de brazos.

### 5.3 Arrepentimiento (Regret)

El arrepentimiento mide la diferencia entre la recompensa esperada de elegir siempre el brazo óptimo y la recompensa esperada de las elecciones del agente:

$\text{Regret} = \sum_{t=1}^{T} \max_i p_i(t) - \sum_{t=1}^{T} p_{a_t+1}(t)$

Valores más bajos de arrepentimiento indican un mejor rendimiento.

### 5.4 Distribución de Recompensas

El entorno visualiza la distribución de recompensas en diferentes entornos usando diagramas de caja (boxplots) y diagramas de violín (violin plots). Estas visualizaciones ayudan a entender:
- La mediana del rendimiento
- La variabilidad en el rendimiento
- La presencia de valores atípicos
- La forma general de la distribución de recompensas

## 6. Pautas de Estrategia

### 6.1 Enfoques Generales

Aquí hay algunos enfoques generales a considerar para la implementación de tus agentes:

1. **Selección Aleatoria**: Elegir brazos aleatoriamente (enfoque de referencia).
2. **Greedy (Codicioso)**: Elegir siempre el brazo con la recompensa estimada más alta.
3. **ε-Greedy**: Casi siempre elegir el mejor brazo, pero explorar ocasionalmente.
4. **UCB (Upper Confidence Bound)**: Elegir brazos basados en estimaciones optimistas de su valor.
5. **Thompson Sampling**: Elegir brazos basados en emparejar probabilidades con distribuciones a posteriori.
6. **Enfoques Bayesianos**: Mantener distribuciones de probabilidad sobre los valores de los brazos.

### 6.2 Consideraciones Específicas del Entorno

#### Banda Fija
- Enfocarse en identificar rápidamente el mejor brazo.
- La exploración se vuelve menos valiosa conforme avanza el juego.
- Con T conocido, se puede planificar un programa decreciente de exploración.

#### Banda Periódica
- Detectar la estructura periódica (k=10).
- Restablecer estimaciones al comienzo de cada período.
- Asignar más exploración al inicio de cada período.

#### Banda Dinámica
- Las observaciones recientes valen más que las antiguas.
- Considerar el uso de una ventana deslizante de observaciones.
- Podría necesitar alta capacidad de respuesta a los cambios.

#### Banda Totalmente Aleatoria
- Estar alerta a cambios repentinos en los patrones de recompensa.
- Equilibrar la persistencia (usar historial) con la adaptabilidad.
- Considerar métodos de detección de cambios.

### 6.3 Consideraciones Específicas de la Información

#### Agente de Información Completa
- Aprovechar el valor conocido p1.
- Enfocarse en estimar p2 con eficiencia.
- Ajustar la estrategia dinámicamente con base en los valores relativos.

#### Agente de Información Parcial
- Similar a información completa, pero más limitado.
- Podría requerir más exploración en ciertos entornos.

#### Agente de Solo Recompensa
- Debe estimar las probabilidades de ambos brazos.
- Necesita lidiar con el horizonte de tiempo desconocido.
- Considerar estrategias adaptativas en el tiempo.

## 7. Conclusión

El problema de Multi-Bandas ofrece un marco fundamental para estudiar la toma de decisiones secuenciales bajo incertidumbre. Los entornos y escenarios de información en este playground brindan un conjunto rico de desafíos que resaltan diferentes aspectos del dilema exploración-explotación.

Al implementar agentes para estos escenarios, obtendrás experiencia práctica con conceptos clave en aprendizaje por refuerzo y teoría de la decisión, y desarrollarás intuición para equilibrar la recolección de información con la maximización de recompensas en diversos contextos.

Mientras trabajas en tus implementaciones, considera cómo se extenderían tus estrategias a:
- Bandas con más de dos brazos.
- Espacios de acción continuos.
- Distribuciones de recompensa no estacionarias con diferentes patrones.
- Bandas contextuales donde se dispone de información adicional.

