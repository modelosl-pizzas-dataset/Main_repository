# Solicitudes de ejemplo a la API (curl / PowerShell)

La API debe estar iniciada en `http://localhost:8000` (ver `README.md`). Los ejemplos usan una pizza real del proyecto.

## Bash / Git Bash / Linux / macOS (curl)

```bash
# Estado de la API
curl http://localhost:8000/health

# Predicción (usa el archivo data/ejemplo_solicitud.json)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud.json

# Predicción escribiendo el JSON en la línea
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"pizza_size":"M","pizza_category":"Classic","pizza_ingredients":"Pepperoni, Mushrooms, Red Onions, Red Peppers, Bacon"}'

# Solicitud inválida: falta pizza_ingredients -> HTTP 422
curl -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"pizza_size":"M","pizza_category":"Classic"}'

# Entrenar una nueva versión (cuerpo vacío) y listar las versiones
curl -X POST http://localhost:8000/train
curl http://localhost:8000/models
```

## PowerShell (Windows)

PowerShell maneja mal las comillas del JSON en `curl.exe`; es más simple usar `Invoke-RestMethod`:

```powershell
# Estado de la API
Invoke-RestMethod http://localhost:8000/health

# Predicción
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -InFile data\ejemplo_solicitud.json

# Solicitud inválida (devuelve error 422: se muestra el mensaje)
try {
  Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"pizza_size":"M","pizza_category":"Classic"}'
} catch { $_.ErrorDetails.Message }

# Entrenar una nueva versión y listar versiones
Invoke-RestMethod -Method Post -Uri http://localhost:8000/train
Invoke-RestMethod http://localhost:8000/models
```

Si prefiere `curl.exe` en PowerShell, use el archivo con `-d "@data/ejemplo_solicitud.json"`.
