NSSM (Non-Sucking Service Manager) example to run the watcher on Windows

1) Download NSSM (https://nssm.cc/) and place `nssm.exe` somewhere accessible (e.g., `C:\tools\nssm\nssm.exe`).

2) Create a service that runs the watcher using the venv Python:

   Open an elevated PowerShell prompt and run (adjust paths as needed):

```powershell
$installDir = 'C:\srv\face-pipeline'
$python = Join-Path $installDir 'venv\Scripts\python.exe'
$nssm = 'C:\tools\nssm\nssm.exe'
& $nssm install FacePipelineService $python "$installDir\src\watcher.py"
# Set service display name, description
& $nssm set FacePipelineService DisplayName "Face Pipeline Watcher"
& $nssm set FacePipelineService Description "Watches photo_input and runs the face detection/anonymization pipeline"
# Start the service
& $nssm start FacePipelineService
```

3) To uninstall the service:

```powershell
& $nssm stop FacePipelineService
& $nssm remove FacePipelineService confirm
```

Notes:
- NSSM is free and commonly used to wrap console applications as services on Windows.
- If the client prefers Scheduled Tasks instead of services, create a task that runs at system startup and runs the venv python command.
