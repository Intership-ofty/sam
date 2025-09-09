# Script PowerShell pour convertir Markdown en PDF
# Nécessite un navigateur installé

$markdownFile = "PRESENTATION_TOWERCO_AIOPS.md"
$htmlFile = "PRESENTATION_TOWERCO_AIOPS.html"
$pdfFile = "PRESENTATION_TOWERCO_AIOPS.pdf"

# Créer un fichier HTML simple
$htmlContent = @"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Présentation Towerco AIOps</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; }
        h3 { color: #7f8c8d; }
        code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        blockquote { border-left: 4px solid #3498db; margin: 0; padding-left: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
"@

# Lire le contenu markdown et le convertir en HTML basique
$markdownContent = Get-Content $markdownFile -Raw

# Conversions markdown basiques
$htmlContent += $markdownContent -replace '^# (.+)$', '<h1>$1</h1>' -replace '^## (.+)$', '<h2>$1</h2>' -replace '^### (.+)$', '<h3>$1</h3>' -replace '^#### (.+)$', '<h4>$1</h4>' -replace '^##### (.+)$', '<h5>$1</h5>' -replace '^###### (.+)$', '<h6>$1</h6>' -replace '\*\*(.+?)\*\*', '<strong>$1</strong>' -replace '\*(.+?)\*', '<em>$1</em>' -replace '`(.+?)`', '<code>$1</code>' -replace '^- (.+)$', '<li>$1</li>' -replace '^\d+\. (.+)$', '<li>$1</li>' -replace '^---$', '<hr>' -replace '\n\n', '</p><p>' -replace '^(.+)$', '<p>$1</p>'

$htmlContent += "</body></html>"

# Sauvegarder le HTML
$htmlContent | Out-File -FilePath $htmlFile -Encoding UTF8

Write-Host "Fichier HTML créé : $htmlFile"
Write-Host "Vous pouvez maintenant :"
Write-Host "1. Ouvrir $htmlFile dans votre navigateur"
Write-Host "2. Imprimer en PDF (Ctrl+P)"
Write-Host "3. Ou utiliser un outil en ligne pour convertir HTML vers PDF"
