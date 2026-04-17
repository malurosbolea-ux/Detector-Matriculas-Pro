# Sistema inteligente de reconocimiento de matrículas (ANPR v2.0)

Este proyecto desarrolla un sistema completo de lectura de matrículas diseñado como un producto de datos funcional. El objetivo no es solo ejecutar un modelo de detección, sino orquestar una arquitectura multi-modelo capaz de resolver retos reales como la variabilidad de iluminación, los ángulos complejos en parkings y el procesamiento de vídeo en tiempo real.

## El origen del proyecto

La práctica surge de un reto técnico en el que la premisa era clara: un sistema de reconocimiento óptico de caracteres (OCR) genérico no es suficiente para un entorno de producción donde existen textos engañosos y condiciones visuales adversas. Por ello, he diseñado un pipeline que separa responsabilidades y optimiza cada etapa del proceso.

## Arquitectura del sistema

Para garantizar la máxima precisión, he aprovechado la capacidad de cómputo de un entorno de alto rendimiento con GPU A100, lo que me ha permitido trabajar con modelos de alta capacidad sin sacrificar la agilidad.

El flujo de datos se divide en cuatro fases críticas:

1. **Detección de vehículos:** Utilizo YOLOv8x para localizar coches, motos, camiones y autobuses. Esto permite aislar la zona de interés y reducir el ruido visual antes de buscar la matrícula.
2. **Detección de matrículas:** Un segundo modelo YOLOv8m, entrenado específicamente con datasets de matrículas, localiza la placa dentro de la región del vehículo.
3. **Preprocesamiento de imagen:** Esta es la pieza clave del éxito del sistema. Mediante OpenCV, aplico filtros de ecualización adaptativa de histograma (CLAHE), binarización de Gauss y eliminación de ruido para que la lectura sea posible incluso en condiciones de sombra o reflejos.
4. **Extracción y validación:** El motor EasyOCR interpreta los caracteres, que posteriormente son normalizados y validados mediante un post-procesado inteligente para corregir confusiones típicas entre letras y números.

## Procesamiento de vídeo y tracking

A diferencia de las imágenes estáticas, el procesamiento de vídeo permite aprovechar la redundancia temporal. He implementado el algoritmo ByteTrack para realizar un seguimiento continuo de cada vehículo. El sistema acumula las lecturas de matrícula a lo largo de los distintos frames y utiliza una votación por mayoría para devolver el resultado con mayor confianza estadística.

## Resultados y rendimiento

En el benchmark de evaluación con imágenes reales de parking, el sistema ha demostrado una robustez notable:

* **Precisión media:** 92.5% de acierto en la lectura de caracteres validada en test.
* **Tiempo de inferencia:** 895 ms por imagen (procesado completo en GPU A100).
* **Efectividad en vídeo:** Identificación y seguimiento exitoso de 14 vehículos en flujo continuo.

## Despliegue en producción

Para transformar este experimento académico en una herramienta accesible, he desplegado una aplicación web profesional utilizando Streamlit. La interfaz sigue una línea de diseño limpia y cuenta con una infraestructura de monitorización externa para garantizar disponibilidad permanente, evitando la hibernación de los servidores.

## Tecnologías utilizadas

* **Detección espacial:** Ultralytics YOLOv8
* **Procesamiento visual:** OpenCV
* **Motor de lectura:** EasyOCR (CNN + RNN)
* **Seguimiento:** ByteTrack
* **Interfaz y nube:** Streamlit Cloud y GitHub

---

## Sobre la autora

Soy María Luisa Ros Bolea, graduada en Comunicación Digital y finalizando mi Máster en Big Data e Inteligencia Artificial en la Universidad CEU San Pablo. Combino mi experiencia técnica en Python y SQL con mi perfil de Marketing & Communications Manager para traducir los aspectos técnicos de los datos en estrategias digitales ejecutables. 

Puedes conocer más sobre mi trabajo o contactar conmigo a través de mis perfiles:

* **Portfolio web:** [malurosbolea-ux.github.io/digital-strategy-portfolio](https://malurosbolea-ux.github.io/digital-strategy-portfolio/)
* **LinkedIn:** [María Luisa Ros Bolea](https://www.linkedin.com/in/mar%C3%ADa-luisa-ros-bolea-400780160/)
* **Instagram:** [@malu_menolu](https://www.instagram.com/malu_menolu/)
* **Email:** [malurosbolea@gmail.com](mailto:malurosbolea@gmail.com)
