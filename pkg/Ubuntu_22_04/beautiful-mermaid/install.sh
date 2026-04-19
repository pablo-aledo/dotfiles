sudo apt-get install -y ca-certificates curl gnupg
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
  | sudo tee /etc/apt/sources.list.d/nodesource.list
sudo apt-get update && sudo apt-get install -y nodejs

mkdir mi-proyecto-mermaid && cd mi-proyecto-mermaid
npm init -y
node -e "const fs=require('fs'); const p=JSON.parse(fs.readFileSync('package.json')); p.type='module'; fs.writeFileSync('package.json', JSON.stringify(p,null,2))"
npm install beautiful-mermaid

cat > diagrama.mjs << 'ENDOFFILE'
import { renderMermaidSVG, THEMES } from 'beautiful-mermaid'
import { writeFileSync } from 'fs'

const diagrama = `
graph TD
  A[Compositor] -->|genera| B[MIDI]
  A -->|define| C[YAML Score]
  B --> D{Análisis}
  C --> D
  D -->|chord detection| E[Chord Table]
  D -->|tension| F[Curva Dramática]
`

const svg = renderMermaidSVG(diagrama, THEMES['tokyo-night'])
const html = `


${svg}

`

writeFileSync('diagrama.html', html)
console.log('Diagrama generado: diagrama.html')
ENDOFFILE

node diagrama.mjs
