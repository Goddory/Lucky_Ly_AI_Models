$ErrorActionPreference = 'Stop'

$taskName = 'Avatar3DPipelineApiAutostart'
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $existingTask) {
	Write-Output "not_found:$taskName"
	exit 0
}

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
Write-Output "deleted:$taskName"
