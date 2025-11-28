# Path to your raw audio folder
$inputRoot  = "C:\Users\laksh\Desktop\Visual Code\Aslinakli\data\audio"
# Path where processed audio will be saved
$outputRoot = "C:\Users\laksh\Desktop\Visual Code\Aslinakli\data\processed_audio"

# target parameters
$targetSR    = 16000       # 16 kHz
$targetDur   = 5           # 5 seconds

Write-Host "[+] Starting audio preprocessing..."

# Loop over language folders (e.g. hindi, marathi, etc.)
Get-ChildItem -Path $inputRoot -Directory | ForEach-Object {
    $lang = $_.Name
    foreach ($cls in @("real","fake")) {
        $inPath  = Join-Path $_.FullName $cls
        $outPath = Join-Path (Join-Path $outputRoot $lang) $cls
        New-Item -Force -ItemType Directory -Path $outPath | Out-Null

        Get-ChildItem -Path $inPath -Filter "*.mp3" | ForEach-Object {
            $inputFile  = $_.FullName
            $fileName   = $_.BaseName + ".wav"
            $outputFile = Join-Path $outPath $fileName

            # ffmpeg convert + resample + trim/pad
            $cmd = "ffmpeg -loglevel error -y -i `"$inputFile`" -ar $targetSR -t $targetDur -af apad `"$outputFile`""
            Invoke-Expression $cmd
            Write-Host "   Processed → $outputFile"
        }
    }
}

Write-Host "[DONE] Preprocessing complete. Processed files in $outputRoot"
