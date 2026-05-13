# KAGGLE_CLOUD_PACKAGE_PLAN

Este documento define la estructura del paquete real que se subirá a Kaggle una vez validado el entorno con el smoke test.

## Componentes del Paquete
- **Runner Congelado**: Versión específica de la lógica de backtesting.
- **Config Micro-probe**: Archivo JSON con parámetros acotados (p.ej. 1 mes de EURUSD).
- **Requirements Mínimos**: Lista de dependencias necesarias en Kaggle.
- **Dataset Reducido**: Archivos Parquet optimizados para la corrida (no raw masivo).
- **Checkpoints**: Lógica para guardar estado cada X iteraciones.
- **Resume Logic**: Capacidad de retomar si la sesión de Kaggle se corta.
- **Output Manifest**: Generación automática de resumen de resultados.

## Restricciones
- **No Raw Data Masiva**: Evitar subir gigabytes; usar solo lo necesario.
- **No venv**: Kaggle ya provee el entorno; instalar solo extras con pip.
- **No .git**: El paquete debe ser aséptico (sin historial de git si se sube como dataset).
- **No Secrets**: Las claves de acceso se manejan vía Kaggle Secrets, no dentro del paquete.
- **No Broker**: El paquete no debe contener lógica de ejecución real.

## Construcción
El paquete real debe crearse fuera del proyecto para evitar contaminar la raíz:
`C:\Users\alera\Desktop\CLOUD_UPLOAD_PACKAGES\`
