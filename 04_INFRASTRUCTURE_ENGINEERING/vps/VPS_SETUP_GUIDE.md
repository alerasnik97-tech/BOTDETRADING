# VPS SETUP GUIDE (Windows)

Esta guía detalla el proceso para configurar el entorno de trading en una VPS Windows.

## 1. Requisitos de Sistema
- **SO:** Windows Server 2019+ o Windows 10/11.
- **CPU:** 2 vCPU mínimo.
- **RAM:** 4 GB mínimo.
- **Disco:** 40 GB SSD.
- **Red:** Conexión estable 24/5.

## 2. Instalación de Software Base
1. **Git:** [Descargar e instalar](https://git-scm.com/).
2. **Python 3.11:** [Descargar](https://www.python.org/downloads/). Asegurarse de marcar "Add Python to PATH".
3. **MetaTrader 5:** Instalar el terminal de tu broker.

## 3. Despliegue del Proyecto
1. Abrir PowerShell y navegar a la carpeta de trabajo.
2. Clonar el repositorio:
   ```powershell
   git clone https://github.com/alerasnik97-tech/bottrading.git
   cd bottrading
   ```
3. Cambiar a la rama segura:
   ```powershell
   git checkout chore/github-clean-sync
   ```

## 4. Configuración del Entorno Python
1. Crear entorno virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Instalar dependencias:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r 04_INFRASTRUCTURE_ENGINEERING/python_environment/requirements.txt
   ```

## 5. Configuración Local (Crítico)
1. Ir a `VPS_READINESS\config_templates\`.
2. Copiar `mt5_vps_config.example.json` a la raíz como `mt5_local_config.json`.
3. Editar `mt5_local_config.json` con tus credenciales de cuenta **DEMO**.
4. **JAMÁS SUBIR ESTE ARCHIVO A GITHUB.**

## 6. Validación
Ejecutar el script de preflight:
```powershell
.\VPS_READINESS\scripts\vps_preflight_check.ps1
```
Si el veredicto es `PASSED`, el sistema está listo para monitoreo demo.
