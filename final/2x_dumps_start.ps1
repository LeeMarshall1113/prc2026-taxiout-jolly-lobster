# Start final/2x_dumps_chain.sh hidden and detached (WMI, ShowWindow = 0), so it survives the session. Safe to run twice.
$repo = "."
$si = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$si.ShowWindow = 0
$cmd = '"C:\Program Files\Git\bin\bash.exe" -lc "cd . && PY=/c/Python314/python.exe bash final/2x_dumps_chain.sh"'
$r = ([wmiclass]"Win32_Process").Create($cmd, $repo, $si)
"started: ReturnValue $($r.ReturnValue) pid $($r.ProcessId)"
